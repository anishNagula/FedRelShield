from dataclasses import dataclass
from typing import List, Optional
import random

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
    ):
        if num_events < 0:
            raise ValueError("num_events must be non-negative")

        if timestamp_step <= 0:
            raise ValueError("timestamp_step must be positive")

        self.seed = seed
        self.num_events = num_events
        self.start_timestamp = start_timestamp
        self.timestamp_step = timestamp_step

    def generate(
        self,
        topology: EnterpriseTopology,
    ) -> List[SecurityEvent]:
        rng = random.Random(self.seed)

        users = self._entities_of_type(topology, "User")
        hosts = self._entities_of_type(topology, "Host")
        servers = self._entities_of_type(topology, "Server")
        processes = self._entities_of_type(topology, "Process")
        files = self._entities_of_type(topology, "File")
        services = self._entities_of_type(topology, "Service")

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
                "Topology is missing entities required for benign generation"
            )

        user_primary_hosts = self._build_user_primary_hosts(
            topology,
            users,
            hosts,
        )

        process_locations = self._build_process_locations(
            topology,
        )

        file_locations = self._build_file_locations(
            topology,
        )

        service_hosts = self._build_service_hosts(
            topology,
        )

        host_peer_sets = self._build_host_peer_sets(
            rng,
            hosts,
            servers,
        )

        relation_generators = [
            (
                0.30,
                lambda: self._authentication_event(
                    rng,
                    users,
                    hosts,
                    servers,
                    user_primary_hosts,
                ),
            ),
            (
                0.25,
                lambda: self._connection_event(
                    rng,
                    hosts,
                    servers,
                    host_peer_sets,
                ),
            ),
            (
                0.20,
                lambda: self._file_access_event(
                    rng,
                    processes,
                    files,
                    process_locations,
                    file_locations,
                ),
            ),
            (
                0.15,
                lambda: self._service_invocation_event(
                    rng,
                    processes,
                    services,
                    process_locations,
                    service_hosts,
                ),
            ),
            (
                0.10,
                lambda: self._process_spawn_event(
                    rng,
                    processes,
                    process_locations,
                ),
            ),
        ]

        events = []

        for event_index in range(self.num_events):
            head, relation, tail = self._sample_event(
                rng,
                relation_generators,
            )

            head_type = topology.entities[head].entity_type
            tail_type = topology.entities[tail].entity_type

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
                    event_id=f"benign_{event_index:06d}",
                    timestamp=timestamp,
                    head=head,
                    relation=relation,
                    tail=tail,
                    label="benign",
                )
            )

        return events

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

    def _build_user_primary_hosts(
        self,
        topology,
        users,
        hosts,
    ):
        admin_hosts = {}

        for edge in topology.edges:
            if edge.relation == "admin_of":
                admin_hosts.setdefault(edge.head, []).append(edge.tail)

        return {
            user: admin_hosts.get(user, [hosts[0]])[0]
            for user in users
        }

    def _build_process_locations(self, topology):
        locations = {}

        for edge in topology.edges:
            if edge.relation == "runs_on":
                locations[edge.head] = edge.tail

        return locations

    def _build_file_locations(self, topology):
        locations = {}

        for edge in topology.edges:
            if edge.relation == "stored_on":
                locations[edge.head] = edge.tail

        return locations

    def _build_service_hosts(self, topology):
        hosts = {}

        for edge in topology.edges:
            if edge.relation == "hosts_service":
                hosts[edge.tail] = edge.head

        return hosts

    def _build_host_peer_sets(
        self,
        rng,
        hosts,
        servers,
    ):
        compute_nodes = hosts + servers
        peer_sets = {}

        for node in compute_nodes:
            candidates = [
                candidate
                for candidate in compute_nodes
                if candidate != node
            ]

            peer_count = min(5, len(candidates))

            peer_sets[node] = rng.sample(
                candidates,
                peer_count,
            )

        return peer_sets

    def _sample_event(
        self,
        rng,
        relation_generators,
    ):
        value = rng.random()
        cumulative_probability = 0.0

        for probability, generator in relation_generators:
            cumulative_probability += probability

            if value < cumulative_probability:
                return generator()

        return relation_generators[-1][1]()

    def _authentication_event(
        self,
        rng,
        users,
        hosts,
        servers,
        user_primary_hosts,
    ):
        user = rng.choice(users)

        if rng.random() < 0.85:
            target = user_primary_hosts[user]
        else:
            target = rng.choice(hosts + servers)

        return user, "authenticates_to", target

    def _connection_event(
        self,
        rng,
        hosts,
        servers,
        host_peer_sets,
    ):
        source = rng.choice(hosts + servers)

        if rng.random() < 0.90:
            target = rng.choice(host_peer_sets[source])
        else:
            candidates = [
                node
                for node in hosts + servers
                if node != source
            ]
            target = rng.choice(candidates)

        return source, "connects_to", target

    def _file_access_event(
        self,
        rng,
        processes,
        files,
        process_locations,
        file_locations,
    ):
        process = rng.choice(processes)
        process_host = process_locations[process]

        local_files = [
            file_entity
            for file_entity in files
            if file_locations[file_entity] == process_host
        ]

        if local_files and rng.random() < 0.85:
            target = rng.choice(local_files)
        else:
            target = rng.choice(files)

        return process, "accesses", target

    def _service_invocation_event(
        self,
        rng,
        processes,
        services,
        process_locations,
        service_hosts,
    ):
        process = rng.choice(processes)
        process_host = process_locations[process]

        local_services = [
            service
            for service in services
            if service_hosts[service] == process_host
        ]

        if local_services and rng.random() < 0.70:
            target = rng.choice(local_services)
        else:
            target = rng.choice(services)

        return process, "invokes", target

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
            if process != parent
            and process_locations[process] == parent_host
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

        return parent, "spawns", child
