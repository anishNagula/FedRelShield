from dataclasses import dataclass, field
from typing import Dict, List
import random

from fedrelshield.data.profiles import (
    EnterpriseProfile,
    ENTERPRISE_A_PROFILE,
)
from fedrelshield.data.schema import (
    ENTERPRISE_A_SCHEMA,
    SecuritySchema,
)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    head: str
    relation: str
    tail: str


@dataclass
class EnterpriseTopology:
    enterprise_id: str
    schema: SecuritySchema
    entities: Dict[str, Entity] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_entity(self, entity: Entity):
        self.schema.validate_entity_type(entity.entity_type)

        if entity.entity_id in self.entities:
            raise ValueError(
                f"Duplicate entity ID: {entity.entity_id}"
            )

        self.entities[entity.entity_id] = entity

    def add_edge(self, edge: Edge):
        if edge.head not in self.entities:
            raise ValueError(
                f"Unknown head entity: {edge.head}"
            )

        if edge.tail not in self.entities:
            raise ValueError(
                f"Unknown tail entity: {edge.tail}"
            )

        head_type = self.entities[edge.head].entity_type
        tail_type = self.entities[edge.tail].entity_type

        self.schema.validate_edge(
            head_type,
            edge.relation,
            tail_type,
        )

        self.edges.append(edge)


class EnterpriseTopologyGenerator:
    def __init__(
        self,
        profile: EnterpriseProfile,
        seed: int = 1024,
        schema: SecuritySchema = ENTERPRISE_A_SCHEMA,
    ):
        profile.validate()

        self.profile = profile
        self.seed = seed
        self.schema = schema

    def generate(self) -> EnterpriseTopology:
        topology = EnterpriseTopology(
            enterprise_id=self.profile.enterprise_id,
            schema=self.schema,
        )

        self._create_entities(topology)
        self._create_structural_edges(topology)

        return topology

    def _create_entities(
        self,
        topology: EnterpriseTopology,
    ):
        for entity_type, count in (
            self.profile.entity_counts.items()
        ):
            self.schema.validate_entity_type(entity_type)

            prefix = entity_type.lower()

            for index in range(count):
                topology.add_entity(
                    Entity(
                        entity_id=(
                            f"{prefix}_{index:03d}"
                        ),
                        entity_type=entity_type,
                    )
                )

    def _create_structural_edges(
        self,
        topology: EnterpriseTopology,
    ):
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
        domain_controllers = self._entities_of_type(
            topology,
            "DomainController",
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

        for node in hosts + servers:
            self._add_edge(
                topology,
                node,
                "member_of",
                rng.choice(domain_controllers),
            )

        for user in users:
            assigned_hosts = set()

            for _ in range(
                self.profile.admin_assignments_per_user
            ):
                available_hosts = [
                    host
                    for host in hosts
                    if host not in assigned_hosts
                ]

                if not available_hosts:
                    break

                host = rng.choice(available_hosts)
                assigned_hosts.add(host)

                self._add_edge(
                    topology,
                    user,
                    "admin_of",
                    host,
                )

        active_compute_nodes = self._select_active_nodes(
            rng,
            hosts,
            servers,
        )

        self._place_entities(
            topology=topology,
            rng=rng,
            entity_ids=processes,
            relation="runs_on",
            active_compute_nodes=active_compute_nodes,
            server_probability=(
                self.profile.process_server_probability
            ),
        )

        self._place_entities(
            topology=topology,
            rng=rng,
            entity_ids=files,
            relation="stored_on",
            active_compute_nodes=active_compute_nodes,
            server_probability=(
                self.profile.file_server_probability
            ),
        )

        service_hosts = self._select_service_hosts(
            rng,
            servers,
        )

        for service in services:
            self._add_edge(
                topology,
                rng.choice(service_hosts),
                "hosts_service",
                service,
            )

    def _select_active_nodes(
        self,
        rng,
        hosts,
        servers,
    ):
        compute_nodes = hosts + servers

        target_count = min(
            self.profile.active_compute_nodes,
            len(compute_nodes),
        )

        server_target = min(
            len(servers),
            round(
                target_count
                * self.profile.process_server_probability
            ),
        )

        host_target = target_count - server_target

        if host_target > len(hosts):
            deficit = host_target - len(hosts)
            host_target = len(hosts)
            server_target = min(
                len(servers),
                server_target + deficit,
            )

        selected = []

        if host_target:
            selected.extend(
                rng.sample(hosts, host_target)
            )

        if server_target:
            selected.extend(
                rng.sample(servers, server_target)
            )

        rng.shuffle(selected)

        return selected

    def _place_entities(
        self,
        topology,
        rng,
        entity_ids,
        relation,
        active_compute_nodes,
        server_probability,
    ):
        active_hosts = [
            node
            for node in active_compute_nodes
            if topology.entities[node].entity_type == "Host"
        ]

        active_servers = [
            node
            for node in active_compute_nodes
            if topology.entities[node].entity_type == "Server"
        ]

        for index, entity_id in enumerate(entity_ids):
            if index < len(active_compute_nodes):
                target = active_compute_nodes[index]
            else:
                prefer_server = (
                    rng.random() < server_probability
                )

                preferred = (
                    active_servers
                    if prefer_server
                    else active_hosts
                )

                fallback = (
                    active_hosts
                    if prefer_server
                    else active_servers
                )

                candidates = preferred or fallback

                target = rng.choice(candidates)

            self._add_edge(
                topology,
                entity_id,
                relation,
                target,
            )

    def _select_service_hosts(
        self,
        rng,
        servers,
    ):
        concentration = (
            self.profile.service_server_concentration
        )

        host_count = max(
            1,
            round(len(servers) * concentration),
        )

        host_count = min(
            host_count,
            len(servers),
        )

        return rng.sample(
            servers,
            host_count,
        )

    def _entities_of_type(
        self,
        topology: EnterpriseTopology,
        entity_type: str,
    ) -> List[str]:
        return [
            entity_id
            for entity_id, entity
            in topology.entities.items()
            if entity.entity_type == entity_type
        ]

    def _add_edge(
        self,
        topology: EnterpriseTopology,
        head: str,
        relation: str,
        tail: str,
    ):
        topology.add_edge(
            Edge(
                head=head,
                relation=relation,
                tail=tail,
            )
        )


class EnterpriseATopologyGenerator(
    EnterpriseTopologyGenerator
):
    def __init__(
        self,
        seed: int = 1024,
        schema: SecuritySchema = ENTERPRISE_A_SCHEMA,
    ):
        super().__init__(
            profile=ENTERPRISE_A_PROFILE,
            seed=seed,
            schema=schema,
        )

