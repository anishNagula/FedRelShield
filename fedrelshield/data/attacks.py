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
            raise ValueError(
                "attack campaign must contain at least one step"
            )

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
            raise ValueError(
                "num_campaigns must be non-negative"
            )

        if timestamp_step <= 0:
            raise ValueError(
                "timestamp_step must be positive"
            )

        if max_attempts_per_campaign <= 0:
            raise ValueError(
                "max_attempts_per_campaign must be positive"
            )

        self.seed = seed
        self.num_campaigns = num_campaigns
        self.start_timestamp = start_timestamp
        self.timestamp_step = timestamp_step

        # Retained for configuration compatibility.
        self.max_attempts_per_campaign = (
            max_attempts_per_campaign
        )

    def inject(self, topology) -> List[AttackCampaign]:
        import random
    
        rng = random.Random(self.seed)
    
        users = self._entities_of_type(
            topology,
            "User",
        )
    
        hosts = self._entities_of_type(
            topology,
            "Host",
        )
    
        servers = self._entities_of_type(
            topology,
            "Server",
        )
    
        processes = self._entities_of_type(
            topology,
            "Process",
        )
    
        services = self._entities_of_type(
            topology,
            "Service",
        )
    
        if not all(
            [
                users,
                hosts,
                servers,
                processes,
                services,
            ]
        ):
            raise ValueError(
                "Topology is missing entities required "
                "for attack injection"
            )
    
        process_locations = self._process_locations(
            topology
        )
    
        service_hosts = self._service_hosts(
            topology
        )
    
        processes_by_node = (
            self._group_processes_by_node(
                process_locations
            )
        )
    
        services_by_server = (
            self._group_services_by_server(
                service_hosts
            )
        )
    
        eligible_targets = [
            server
            for server in servers
            if processes_by_node.get(server)
            and services_by_server.get(server)
        ]
    
        if not eligible_targets:
            raise ValueError(
                "Topology has no server with both "
                "a local process and a hosted service"
            )
    
        campaign_candidates = (
            self._build_campaign_candidates(
                users=users,
                processes=processes,
                eligible_targets=eligible_targets,
                process_locations=process_locations,
                processes_by_node=processes_by_node,
                services_by_server=services_by_server,
            )
        )
    
        if not campaign_candidates:
            raise RuntimeError(
                "No feasible attack campaign candidates"
            )
    
        #
        # Campaigns are sampled without duplicate complete campaign
        # parameterizations.
        #
        # Individual attack triples may occur in multiple campaigns.
        # This is intentional: shared infrastructure, accounts,
        # processes, and services are realistic in enterprise attack
        # activity.
        #
        # Link-prediction split disjointness is enforced later by
        # KGExporter.
        #
    
        if self.num_campaigns > len(campaign_candidates):
            raise RuntimeError(
                "Requested more attack campaigns than "
                "available unique campaign candidates: "
                f"requested={self.num_campaigns}, "
                f"available={len(campaign_candidates)}"
            )
    
        selected_candidates = rng.sample(
            campaign_candidates,
            self.num_campaigns,
        )
    
        campaigns = []
    
        for campaign_index, candidate in enumerate(
            selected_candidates
        ):
            (
                source_process,
                compromised_user,
                target_node,
                remote_process,
                target_service,
                _,
            ) = candidate
    
            campaign_id = (
                f"campaign_{campaign_index:03d}"
            )
    
            initial_timestamp = (
                self.start_timestamp
                + campaign_index
                * self.STEPS_PER_CAMPAIGN
                * self.timestamp_step
            )
    
            campaign = self._build_campaign(
                topology=topology,
                campaign_id=campaign_id,
                initial_timestamp=initial_timestamp,
                source_process=source_process,
                compromised_user=compromised_user,
                target_node=target_node,
                remote_process=remote_process,
                target_service=target_service,
            )
    
            campaigns.append(campaign)
    
        return campaigns

    def _build_campaign_candidates(
        self,
        users,
        processes,
        eligible_targets,
        process_locations,
        processes_by_node,
        services_by_server,
    ):
        candidates = []

        for source_process in processes:
            source_node = process_locations[
                source_process
            ]

            for compromised_user in users:
                for target_node in eligible_targets:
                    if target_node == source_node:
                        continue

                    for remote_process in (
                        processes_by_node[target_node]
                    ):
                        for target_service in (
                            services_by_server[target_node]
                        ):
                            triples = (
                                (
                                    source_process,
                                    "uses_credential",
                                    compromised_user,
                                ),
                                (
                                    compromised_user,
                                    "authenticates_to",
                                    target_node,
                                ),
                                (
                                    source_node,
                                    "connects_to",
                                    target_node,
                                ),
                                (
                                    remote_process,
                                    "executes_on",
                                    target_node,
                                ),
                                (
                                    remote_process,
                                    "invokes",
                                    target_service,
                                ),
                            )

                            if (
                                len(set(triples))
                                != self.STEPS_PER_CAMPAIGN
                            ):
                                continue

                            candidates.append(
                                (
                                    source_process,
                                    compromised_user,
                                    target_node,
                                    remote_process,
                                    target_service,
                                    frozenset(triples),
                                )
                            )

        return candidates

    def _build_campaign(
        self,
        topology,
        campaign_id,
        initial_timestamp,
        source_process,
        compromised_user,
        target_node,
        remote_process,
        target_service,
    ) -> AttackCampaign:
        source_node = self._process_locations(
            topology
        )[source_process]

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

            head_type = (
                topology.entities[head].entity_type
            )

            tail_type = (
                topology.entities[tail].entity_type
            )

            topology.schema.validate_edge(
                head_type,
                relation,
                tail_type,
            )

            event = SecurityEvent(
                event_id=(
                    f"attack_{campaign_id}_"
                    f"{step_index:03d}"
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
            for entity_id, entity
            in topology.entities.items()
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

