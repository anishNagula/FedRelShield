from dataclasses import dataclass
from typing import FrozenSet, Tuple


RelationSignature = Tuple[str, str, str]


@dataclass(frozen=True)
class SecuritySchema:
    entity_types: FrozenSet[str]
    relation_types: FrozenSet[str]
    relation_signatures: FrozenSet[RelationSignature]

    def validate_entity_type(self, entity_type):
        if entity_type not in self.entity_types:
            raise ValueError(
                f"Unknown entity type: {entity_type}"
            )

    def validate_relation_type(self, relation_type):
        if relation_type not in self.relation_types:
            raise ValueError(
                f"Unknown relation type: {relation_type}"
            )

    def validate_edge(
        self,
        head_type,
        relation_type,
        tail_type,
    ):
        self.validate_entity_type(head_type)
        self.validate_relation_type(relation_type)
        self.validate_entity_type(tail_type)

        signature = (
            head_type,
            relation_type,
            tail_type,
        )

        if signature not in self.relation_signatures:
            raise ValueError(
                "Invalid relation signature: "
                f"{head_type} --{relation_type}--> {tail_type}"
            )


ENTERPRISE_A_SCHEMA = SecuritySchema(
    entity_types=frozenset(
        {
            "User",
            "Host",
            "Server",
            "DomainController",
            "Process",
            "File",
            "Service",
        }
    ),
    relation_types=frozenset(
        {
            "member_of",
            "admin_of",
            "runs_on",
            "stored_on",
            "authenticates_to",
            "connects_to",
            "spawns",
            "accesses",
            "hosts_service",
            "invokes",
            "uses_credential",
            "executes_on",
        }
    ),
    relation_signatures=frozenset(
        {
            ("User", "admin_of", "Host"),
            ("User", "authenticates_to", "Host"),
            ("User", "authenticates_to", "Server"),
            ("User", "authenticates_to", "DomainController"),

            ("Host", "member_of", "DomainController"),
            ("Server", "member_of", "DomainController"),

            ("Process", "runs_on", "Host"),
            ("Process", "runs_on", "Server"),

            ("File", "stored_on", "Host"),
            ("File", "stored_on", "Server"),

            ("Server", "hosts_service", "Service"),

            ("Host", "connects_to", "Host"),
            ("Host", "connects_to", "Server"),
            ("Host", "connects_to", "DomainController"),
            ("Server", "connects_to", "Host"),
            ("Server", "connects_to", "Server"),
            ("Server", "connects_to", "DomainController"),

            ("Process", "spawns", "Process"),
            ("Process", "accesses", "File"),
            ("Process", "invokes", "Service"),

            ("Process", "uses_credential", "User"),

            ("Process", "executes_on", "Host"),
            ("Process", "executes_on", "Server"),
        }
    ),
)
