from dataclasses import dataclass, field
from typing import Dict, List
import random

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

        # ---------------------------------------------------------
        # Active Directory membership
        # ---------------------------------------------------------

        # Every workstation belongs to one AD domain controller.
        for host in hosts:
            self._add_edge(
                topology,
                host,
                "member_of",
                rng.choice(domain_controllers),
            )

        # Every server belongs to one AD domain controller.
        for server in servers:
            self._add_edge(
                topology,
                server,
                "member_of",
                rng.choice(domain_controllers),
            )

        # ---------------------------------------------------------
        # User privileges
        # ---------------------------------------------------------

        # Each user receives administrative privileges on one host.
        #
        # This remains intentionally simple for Enterprise A.
        # Later enterprise profiles will control privilege density
        # and user-to-machine assignment independently.
        for user in users:
            self._add_edge(
                topology,
                user,
                "admin_of",
                rng.choice(hosts),
            )

        # ---------------------------------------------------------
        # Active behavioral footprint
        # ---------------------------------------------------------

        compute_nodes = hosts + servers

        # Not every machine necessarily produces process/file
        # telemetry during one observation window.
        #
        # We therefore select a deterministic, seeded subset of
        # active compute nodes.
        #
        # Both processes and files are placed on these nodes so
        # topology-conditioned local file access is structurally
        # possible.

        num_active_compute_nodes = min(
            30,
            len(compute_nodes),
        )

        active_compute_nodes = rng.sample(
            compute_nodes,
            num_active_compute_nodes,
        )

        # ---------------------------------------------------------
        # Process placement
        # ---------------------------------------------------------

        # Guarantee every active node contains at least one process.
        #
        # Remaining processes are randomly distributed across the
        # same active footprint.

        for process_index, process in enumerate(processes):
            if process_index < len(active_compute_nodes):
                target_node = active_compute_nodes[process_index]
            else:
                target_node = rng.choice(active_compute_nodes)

            self._add_edge(
                topology,
                process,
                "runs_on",
                target_node,
            )

        # ---------------------------------------------------------
        # File placement
        # ---------------------------------------------------------

        # Guarantee every active node contains at least one file.
        #
        # This ensures every process location has at least one
        # local file available for topology-conditioned benign
        # access generation.

        for file_index, file_entity in enumerate(files):
            if file_index < len(active_compute_nodes):
                target_node = active_compute_nodes[file_index]
            else:
                target_node = rng.choice(active_compute_nodes)

            self._add_edge(
                topology,
                file_entity,
                "stored_on",
                target_node,
            )

        # ---------------------------------------------------------
        # Service placement
        # ---------------------------------------------------------

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
