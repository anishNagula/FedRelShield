from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from fedrelshield.data.profiles import (
    BENIGN_RELATIONS,
    EnterpriseProfile,
    get_enterprise_profile,
)
from fedrelshield.data.topology import EnterpriseTopology


@dataclass(frozen=True)
class SecurityEvent:
    event_id: str
    timestamp: int
    head: str
    relation: str
    tail: str
    label: str
    campaign_id: Optional[str] = None


class BenignEventGenerator:
    def __init__(
        self,
        seed: int = 1024,
        num_events: int = 2000,
        start_timestamp: int = 0,
        timestamp_step: int = 1,
        profile: Optional[EnterpriseProfile] = None,
    ):
        if num_events < 0:
            raise ValueError(
                "num_events must be non-negative"
            )

        if timestamp_step <= 0:
            raise ValueError(
                "timestamp_step must be positive"
            )

        if profile is not None:
            profile.validate()

        self.seed = seed
        self.num_events = num_events
        self.start_timestamp = start_timestamp
        self.timestamp_step = timestamp_step
        self.profile = profile

    def generate(
        self,
        topology: EnterpriseTopology,
    ) -> List[SecurityEvent]:
        profile = self._resolve_profile(topology)
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
        files = self._entities_of_type(
            topology,
            "File",
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
                files,
                services,
            ]
        ):
            raise ValueError(
                "Topology is missing entities required "
                "for benign generation"
            )

        user_primary_hosts = (
            self._build_user_primary_hosts(
                topology,
                users,
                hosts,
            )
        )

        process_locations = (
            self._build_process_locations(topology)
        )

        file_locations = (
            self._build_file_locations(topology)
        )

        service_hosts = (
            self._build_service_hosts(topology)
        )

        connection_hotspots = (
            self._select_connection_hotspots(
                rng,
                hosts,
                servers,
                profile,
            )
        )

        host_peer_sets = self._build_host_peer_sets(
            rng,
            hosts,
            servers,
            connection_hotspots,
            profile,
        )

        relation_generators = {
            "authenticates_to": (
                lambda: self._authentication_event(
                    rng,
                    users,
                    hosts,
                    servers,
                    user_primary_hosts,
                    profile,
                )
            ),
            "connects_to": (
                lambda: self._connection_event(
                    rng,
                    hosts,
                    servers,
                    host_peer_sets,
                )
            ),
            "spawns": (
                lambda: self._process_spawn_event(
                    rng,
                    processes,
                    process_locations,
                )
            ),
            "accesses": (
                lambda: self._file_access_event(
                    rng,
                    processes,
                    files,
                    process_locations,
                    file_locations,
                    profile,
                )
            ),
            "invokes": (
                lambda: self._service_invocation_event(
                    rng,
                    processes,
                    services,
                    process_locations,
                    service_hosts,
                    profile,
                )
            ),
        }

        relation_names = list(BENIGN_RELATIONS)

        relation_weights = [
            profile.benign_relation_weights[relation]
            for relation in relation_names
        ]

        events = []

        for event_index in range(self.num_events):
            relation = rng.choices(
                relation_names,
                weights=relation_weights,
                k=1,
            )[0]

            head, sampled_relation, tail = (
                relation_generators[relation]()
            )

            if sampled_relation != relation:
                raise RuntimeError(
                    "Benign relation generator returned "
                    "an unexpected relation"
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

            timestamp = (
                self.start_timestamp
                + event_index * self.timestamp_step
            )

            events.append(
                SecurityEvent(
                    event_id=(
                        f"benign_{event_index:06d}"
                    ),
                    timestamp=timestamp,
                    head=head,
                    relation=relation,
                    tail=tail,
                    label="benign",
                )
            )

        return events

    def _resolve_profile(
        self,
        topology: EnterpriseTopology,
    ) -> EnterpriseProfile:
        if self.profile is not None:
            if (
                self.profile.enterprise_id
                != topology.enterprise_id
            ):
                raise ValueError(
                    "Generator profile does not match "
                    "topology enterprise_id"
                )

            return self.profile

        return get_enterprise_profile(
            topology.enterprise_id
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

    def _build_user_primary_hosts(
        self,
        topology,
        users,
        hosts,
    ):
        admin_hosts: Dict[str, List[str]] = {}

        for edge in topology.edges:
            if edge.relation == "admin_of":
                admin_hosts.setdefault(
                    edge.head,
                    [],
                ).append(edge.tail)

        return {
            user: (
                sorted(admin_hosts[user])[0]
                if user in admin_hosts
                else hosts[
                    self._stable_user_host_index(
                        user,
                        len(hosts),
                    )
                ]
            )
            for user in users
        }

    def _stable_user_host_index(
        self,
        user: str,
        host_count: int,
    ) -> int:
        try:
            user_index = int(
                user.rsplit("_", 1)[1]
            )
        except (IndexError, ValueError):
            user_index = sum(
                ord(character)
                for character in user
            )

        return user_index % host_count

    def _build_process_locations(
        self,
        topology,
    ):
        return {
            edge.head: edge.tail
            for edge in topology.edges
            if edge.relation == "runs_on"
        }

    def _build_file_locations(
        self,
        topology,
    ):
        return {
            edge.head: edge.tail
            for edge in topology.edges
            if edge.relation == "stored_on"
        }

    def _build_service_hosts(
        self,
        topology,
    ):
        return {
            edge.tail: edge.head
            for edge in topology.edges
            if edge.relation == "hosts_service"
        }

    def _select_connection_hotspots(
        self,
        rng,
        hosts,
        servers,
        profile,
    ):
        compute_nodes = hosts + servers

        hotspot_count = max(
            1,
            round(
                len(compute_nodes)
                * profile.connection_hotspot_fraction
            ),
        )

        hotspot_count = min(
            hotspot_count,
            len(compute_nodes),
        )

        return rng.sample(
            compute_nodes,
            hotspot_count,
        )

    def _build_host_peer_sets(
        self,
        rng,
        hosts,
        servers,
        connection_hotspots,
        profile,
    ):
        compute_nodes = hosts + servers
        peer_sets = {}

        target_peer_count = max(
            1,
            round(
                len(compute_nodes)
                * profile.connection_hotspot_fraction
            ),
        )

        target_peer_count = min(
            target_peer_count,
            len(compute_nodes) - 1,
        )

        for node in compute_nodes:
            candidates = [
                candidate
                for candidate in compute_nodes
                if candidate != node
            ]

            hotspot_candidates = [
                candidate
                for candidate in connection_hotspots
                if candidate != node
            ]

            selected = []

            if hotspot_candidates:
                hotspot_target = min(
                    target_peer_count,
                    len(hotspot_candidates),
                )

                selected.extend(
                    rng.sample(
                        hotspot_candidates,
                        hotspot_target,
                    )
                )

            remaining_candidates = [
                candidate
                for candidate in candidates
                if candidate not in selected
            ]

            remaining_target = (
                target_peer_count - len(selected)
            )

            if remaining_target > 0:
                selected.extend(
                    rng.sample(
                        remaining_candidates,
                        min(
                            remaining_target,
                            len(remaining_candidates),
                        ),
                    )
                )

            peer_sets[node] = selected

        return peer_sets

    def _authentication_event(
        self,
        rng,
        users,
        hosts,
        servers,
        user_primary_hosts,
        profile,
    ):
        user = rng.choice(users)

        if (
            rng.random()
            < profile.authentication_primary_probability
        ):
            target = user_primary_hosts[user]
        else:
            target = rng.choice(
                hosts + servers
            )

        return (
            user,
            "authenticates_to",
            target,
        )

    def _connection_event(
        self,
        rng,
        hosts,
        servers,
        host_peer_sets,
    ):
        source = rng.choice(
            hosts + servers
        )

        peers = host_peer_sets[source]

        if peers:
            target = rng.choice(peers)
        else:
            candidates = [
                node
                for node in hosts + servers
                if node != source
            ]

            target = rng.choice(candidates)

        return (
            source,
            "connects_to",
            target,
        )

    def _file_access_event(
        self,
        rng,
        processes,
        files,
        process_locations,
        file_locations,
        profile,
    ):
        process = rng.choice(processes)
        process_host = process_locations[process]

        local_files = [
            file_entity
            for file_entity in files
            if (
                file_locations[file_entity]
                == process_host
            )
        ]

        if (
            local_files
            and rng.random()
            < profile.local_file_access_probability
        ):
            target = rng.choice(local_files)
        else:
            target = rng.choice(files)

        return (
            process,
            "accesses",
            target,
        )

    def _service_invocation_event(
        self,
        rng,
        processes,
        services,
        process_locations,
        service_hosts,
        profile,
    ):
        process = rng.choice(processes)
        process_host = process_locations[process]

        local_services = [
            service
            for service in services
            if (
                service_hosts[service]
                == process_host
            )
        ]

        if (
            local_services
            and rng.random()
            < profile.service_locality_probability
        ):
            target = rng.choice(local_services)
        else:
            target = rng.choice(services)

        return (
            process,
            "invokes",
            target,
        )

    def _process_spawn_event(
        self,
        rng,
        processes,
        process_locations,
    ):
        parent = rng.choice(processes)
        parent_host = process_locations[parent]

        local_processes = [
            process
            for process in processes
            if (
                process != parent
                and process_locations[process]
                == parent_host
            )
        ]

        if local_processes:
            child = rng.choice(local_processes)
        else:
            candidates = [
                process
                for process in processes
                if process != parent
            ]

            child = rng.choice(candidates)

        return (
            parent,
            "spawns",
            child,
        )
