from dataclasses import dataclass, field
from typing import Dict, List

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


class EnterpriseATopologyGenerator:
    def __init__(
        self,
        seed: int = 1024,
        schema: SecuritySchema = ENTERPRISE_A_SCHEMA,
    ):
        self.seed = seed
        self.schema = schema

    def generate(self) -> EnterpriseTopology:
        topology = EnterpriseTopology(
            enterprise_id="enterprise_a",
            schema=self.schema,
        )

        self._create_entities(topology)
        self._create_structural_edges(topology)

        return topology

    def _create_entities(
        self,
        topology: EnterpriseTopology,
    ):
        entity_counts = {
            "User": 40,
            "Host": 60,
            "Server": 15,
            "DomainController": 2,
            "Process": 80,
            "File": 40,
            "Service": 20,
        }

        for entity_type, count in entity_counts.items():
            self.schema.validate_entity_type(entity_type)

            prefix = entity_type.lower()

            for index in range(count):
                entity_id = f"{prefix}_{index:03d}"

                topology.add_entity(
                    Entity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                    )
                )

    def _create_structural_edges(
        self,
        topology: EnterpriseTopology,
    ):
        import random

        rng = random.Random(self.seed)

        users = self._entities_of_type(topology, "User")
        hosts = self._entities_of_type(topology, "Host")
        servers = self._entities_of_type(topology, "Server")
        domain_controllers = self._entities_of_type(
            topology,
            "DomainController",
        )
        processes = self._entities_of_type(topology, "Process")
        files = self._entities_of_type(topology, "File")
        services = self._entities_of_type(topology, "Service")

        # Every workstation and server belongs to one AD domain controller.
        for host in hosts:
            self._add_edge(
                topology,
                host,
                "member_of",
                rng.choice(domain_controllers),
            )

        for server in servers:
            self._add_edge(
                topology,
                server,
                "member_of",
                rng.choice(domain_controllers),
            )

        # Users receive administrative privileges on a sparse subset of hosts.
        # We keep one assignment per user for now; later enterprise profiles
        # will control privilege density.
        for user in users:
            self._add_edge(
                topology,
                user,
                "admin_of",
                rng.choice(hosts),
            )

        # Processes are placed across workstations and servers.
        compute_nodes = hosts + servers

        for process in processes:
            self._add_edge(
                topology,
                process,
                "runs_on",
                rng.choice(compute_nodes),
            )

        # Files are stored across workstations and servers.
        for file_entity in files:
           self._add_edge(
                topology,
                file_entity,
                "stored_on",
                rng.choice(compute_nodes),
            )

        # Services are hosted by servers.
        for service in services:
            self._add_edge(
                topology,
                rng.choice(servers),
                "hosts_service",
                service,
            )

    def _entities_of_type(
        self,
        topology: EnterpriseTopology,
        entity_type: str,
    ) -> List[str]:
        return [
            entity_id
            for entity_id, entity in topology.entities.items()
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
