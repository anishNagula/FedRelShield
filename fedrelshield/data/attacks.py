from dataclasses import dataclass
from typing import List

from fedrelshield.data.benign import SecurityEvent


@dataclass(frozen=True)
class AttackStep:
    step_index: int
    technique: str
    event: SecurityEvent


@dataclass(frozen=True)
class AttackCampaign:
    campaign_id: str
    initial_timestamp: int
    steps: List[AttackStep]

    def __post_init__(self):
        if not self.campaign_id:
            raise ValueError("campaign_id must be non-empty")

        if not self.steps:
            raise ValueError("attack campaign must contain at least one step")

        expected_indices = list(range(len(self.steps)))

        actual_indices = [
            step.step_index
            for step in self.steps
        ]

        if actual_indices != expected_indices:
            raise ValueError(
                "attack campaign steps must have contiguous "
                "zero-based step indices"
            )

        timestamps = [
            step.event.timestamp
            for step in self.steps
        ]

        if timestamps != sorted(timestamps):
            raise ValueError(
                "attack campaign events must be ordered by timestamp"
            )

        if timestamps[0] != self.initial_timestamp:
            raise ValueError(
                "initial_timestamp must match the first attack event"
            )

        for step in self.steps:
            if step.event.label != "attack":
                raise ValueError(
                    "attack campaign events must have label='attack'"
                )

            if step.event.campaign_id != self.campaign_id:
                raise ValueError(
                    "attack event campaign_id does not match campaign"
                )

    @property
    def events(self) -> List[SecurityEvent]:
        return [
            step.event
            for step in self.steps
        ]

    @property
    def final_timestamp(self) -> int:
        return self.steps[-1].event.timestamp


class AttackInjector:
    STEPS_PER_CAMPAIGN = 5

    def __init__(
        self,
        seed: int = 1024,
        num_campaigns: int = 12,
        start_timestamp: int = 2000,
        timestamp_step: int = 1,
        max_attempts_per_campaign: int = 1000,
    ):
        if num_campaigns < 0:
            raise ValueError("num_campaigns must be non-negative")

        if timestamp_step <= 0:
            raise ValueError("timestamp_step must be positive")

        if max_attempts_per_campaign <= 0:
            raise ValueError(
                "max_attempts_per_campaign must be positive"
            )

        self.seed = seed
        self.num_campaigns = num_campaigns
        self.start_timestamp = start_timestamp
        self.timestamp_step = timestamp_step
        self.max_attempts_per_campaign = max_attempts_per_campaign

    def inject(self, topology) -> List[AttackCampaign]:
        import random

        rng = random.Random(self.seed)

        users = self._entities_of_type(topology, "User")
        hosts = self._entities_of_type(topology, "Host")
        servers = self._entities_of_type(topology, "Server")
        processes = self._entities_of_type(topology, "Process")
        services = self._entities_of_type(topology, "Service")

        process_locations = self._process_locations(topology)
        service_hosts = self._service_hosts(topology)

        if not all([users, hosts, servers, processes, services]):
            raise ValueError(
                "Topology is missing entities required for attack injection"
            )

        processes_by_node = self._group_processes_by_node(
            process_locations,
        )

        services_by_server = self._group_services_by_server(
            service_hosts,
        )

        eligible_targets = [
            server
            for server in servers
            if processes_by_node.get(server)
            and services_by_server.get(server)
        ]

        if not eligible_targets:
            raise ValueError(
                "Topology has no server with both a local process "
                "and a hosted service"
            )

        campaigns = []

        # Exact attack triples already used by accepted campaigns.
        #
        # Entities may be reused across campaigns, but the same
        # (head, relation, tail) attack fact may not appear twice.
        used_attack_triples = set()

        for campaign_index in range(self.num_campaigns):
            campaign_id = f"campaign_{campaign_index:03d}"

            initial_timestamp = (
                self.start_timestamp
                + campaign_index
                * self.STEPS_PER_CAMPAIGN
                * self.timestamp_step
            )

            campaign = None

            for _ in range(self.max_attempts_per_campaign):
                candidate = self._generate_candidate_campaign(
                    topology=topology,
                    rng=rng,
                    campaign_id=campaign_id,
                    initial_timestamp=initial_timestamp,
                    users=users,
                    processes=processes,
                    eligible_targets=eligible_targets,
                    process_locations=process_locations,
                    processes_by_node=processes_by_node,
                    services_by_server=services_by_server,
                )

                candidate_triples = {
                    (
                        event.head,
                        event.relation,
                        event.tail,
                    )
                    for event in candidate.events
                }

                # A campaign should not contain duplicate triples
                # internally.
                if len(candidate_triples) != len(candidate.events):
                    continue

                # Reject candidates that reuse any exact attack
                # triple from an earlier accepted campaign.
                if candidate_triples & used_attack_triples:
                    continue

                campaign = candidate

                used_attack_triples.update(
                    candidate_triples
                )

                break

            if campaign is None:
                raise RuntimeError(
                    "Unable to generate a triple-disjoint attack "
                    f"campaign after {self.max_attempts_per_campaign} "
                    f"attempts: {campaign_id}"
                )

            campaigns.append(campaign)

        return campaigns

    def _generate_candidate_campaign(
        self,
        topology,
        rng,
        campaign_id,
        initial_timestamp,
        users,
        processes,
        eligible_targets,
        process_locations,
        processes_by_node,
        services_by_server,
    ) -> AttackCampaign:
        source_process = rng.choice(processes)
        source_node = process_locations[source_process]

        compromised_user = rng.choice(users)

        target_candidates = [
            server
            for server in eligible_targets
            if server != source_node
        ]

        if not target_candidates:
            raise ValueError(
                "No eligible attack target distinct from source node"
            )

        target_node = rng.choice(target_candidates)

        target_service = rng.choice(
            services_by_server[target_node]
        )

        remote_process = rng.choice(
            processes_by_node[target_node]
        )

        event_specs = [
            (
                source_process,
                "uses_credential",
                compromised_user,
                "credential_access",
            ),
            (
                compromised_user,
                "authenticates_to",
                target_node,
                "remote_authentication",
            ),
            (
                source_node,
                "connects_to",
                target_node,
                "lateral_connection",
            ),
            (
                remote_process,
                "executes_on",
                target_node,
                "remote_execution",
            ),
            (
                remote_process,
                "invokes",
                target_service,
                "service_access",
            ),
        ]

        steps = []

        for step_index, (
            head,
            relation,
            tail,
            technique,
        ) in enumerate(event_specs):
            timestamp = (
                initial_timestamp
                + step_index * self.timestamp_step
            )

            head_type = topology.entities[head].entity_type
            tail_type = topology.entities[tail].entity_type

            topology.schema.validate_edge(
                head_type,
                relation,
                tail_type,
            )

            event = SecurityEvent(
                event_id=(
                    f"attack_{campaign_id}_{step_index:03d}"
                ),
                timestamp=timestamp,
                head=head,
                relation=relation,
                tail=tail,
                label="attack",
                campaign_id=campaign_id,
            )

            steps.append(
                AttackStep(
                    step_index=step_index,
                    technique=technique,
                    event=event,
                )
            )

        return AttackCampaign(
            campaign_id=campaign_id,
            initial_timestamp=initial_timestamp,
            steps=steps,
        )

    def _entities_of_type(
        self,
        topology,
        entity_type,
    ):
        return [
            entity_id
            for entity_id, entity in topology.entities.items()
            if entity.entity_type == entity_type
        ]

    def _process_locations(
        self,
        topology,
    ):
        return {
            edge.head: edge.tail
            for edge in topology.edges
            if edge.relation == "runs_on"
        }

    def _service_hosts(
        self,
        topology,
    ):
        return {
            edge.tail: edge.head
            for edge in topology.edges
            if edge.relation == "hosts_service"
        }

    def _group_processes_by_node(
        self,
        process_locations,
    ):
        processes_by_node = {}

        for process, node in process_locations.items():
            processes_by_node.setdefault(
                node,
                [],
            ).append(process)

        return processes_by_node

    def _group_services_by_server(
        self,
        service_hosts,
    ):
        services_by_server = {}

        for service, server in service_hosts.items():
            services_by_server.setdefault(
                server,
                [],
            ).append(service)

        return services_by_server
