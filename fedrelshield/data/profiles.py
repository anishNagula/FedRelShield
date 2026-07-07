from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class EnterpriseProfile:
    enterprise_id: str
    profile_name: str

    entity_counts: Dict[str, int]

    active_compute_nodes: int
    admin_assignments_per_user: int

    process_server_probability: float
    file_server_probability: float

    service_server_concentration: float

    benign_relation_weights: Dict[str, float]

    authentication_primary_probability: float
    local_file_access_probability: float
    connection_hotspot_fraction: float
    service_locality_probability: float

    attack_technique_weights: Dict[str, float]

    def validate(self):
        if not self.enterprise_id:
            raise ValueError("enterprise_id must be non-empty")

        if not self.profile_name:
            raise ValueError("profile_name must be non-empty")

        required_entity_types = {
            "User",
            "Host",
            "Server",
            "DomainController",
            "Process",
            "File",
            "Service",
        }

        if set(self.entity_counts) != required_entity_types:
            raise ValueError(
                "entity_counts must define exactly: "
                f"{sorted(required_entity_types)}"
            )

        if any(
            count <= 0
            for count in self.entity_counts.values()
        ):
            raise ValueError(
                "all entity counts must be positive"
            )

        compute_nodes = (
            self.entity_counts["Host"]
            + self.entity_counts["Server"]
        )

        if not (
            1
            <= self.active_compute_nodes
            <= compute_nodes
        ):
            raise ValueError(
                "active_compute_nodes must be within "
                "the available compute-node count"
            )

        if self.admin_assignments_per_user < 0:
            raise ValueError(
                "admin_assignments_per_user must be non-negative"
            )

        probability_fields = {
            "process_server_probability":
                self.process_server_probability,
            "file_server_probability":
                self.file_server_probability,
            "service_server_concentration":
                self.service_server_concentration,
            "authentication_primary_probability":
                self.authentication_primary_probability,
            "local_file_access_probability":
                self.local_file_access_probability,
            "connection_hotspot_fraction":
                self.connection_hotspot_fraction,
            "service_locality_probability":
                self.service_locality_probability,
        }

        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1]"
                )

        self._validate_weights(
            "benign_relation_weights",
            self.benign_relation_weights,
        )

        self._validate_weights(
            "attack_technique_weights",
            self.attack_technique_weights,
        )

    @staticmethod
    def _validate_weights(
        name: str,
        weights: Dict[str, float],
    ):
        if not weights:
            raise ValueError(f"{name} must be non-empty")

        if any(value < 0 for value in weights.values()):
            raise ValueError(
                f"{name} cannot contain negative weights"
            )

        if sum(weights.values()) <= 0:
            raise ValueError(
                f"{name} must contain positive total weight"
            )


BENIGN_RELATIONS = (
    "authenticates_to",
    "connects_to",
    "spawns",
    "accesses",
    "invokes",
)


ATTACK_TECHNIQUES = (
    "credential_access",
    "remote_authentication",
    "lateral_connection",
    "remote_execution",
    "service_execution",
)


ENTERPRISE_A_PROFILE = EnterpriseProfile(
    enterprise_id="enterprise_a",
    profile_name="windows_active_directory",
    entity_counts={
        "User": 40,
        "Host": 60,
        "Server": 15,
        "DomainController": 2,
        "Process": 80,
        "File": 40,
        "Service": 20,
    },
    active_compute_nodes=30,
    admin_assignments_per_user=1,
    process_server_probability=0.20,
    file_server_probability=0.25,
    service_server_concentration=0.45,
    benign_relation_weights={
        "authenticates_to": 0.30,
        "connects_to": 0.25,
        "spawns": 0.12,
        "accesses": 0.20,
        "invokes": 0.13,
    },
    authentication_primary_probability=0.85,
    local_file_access_probability=0.86,
    connection_hotspot_fraction=0.08,
    service_locality_probability=0.70,
    attack_technique_weights={
        "credential_access": 0.30,
        "remote_authentication": 0.25,
        "lateral_connection": 0.20,
        "remote_execution": 0.15,
        "service_execution": 0.10,
    },
)


ENTERPRISE_B_PROFILE = EnterpriseProfile(
    enterprise_id="enterprise_b",
    profile_name="cloud_native_services",
    entity_counts={
        "User": 24,
        "Host": 24,
        "Server": 55,
        "DomainController": 2,
        "Process": 150,
        "File": 30,
        "Service": 90,
    },
    active_compute_nodes=60,
    admin_assignments_per_user=0,
    process_server_probability=0.85,
    file_server_probability=0.80,
    service_server_concentration=0.18,
    benign_relation_weights={
        "authenticates_to": 0.08,
        "connects_to": 0.28,
        "spawns": 0.12,
        "accesses": 0.07,
        "invokes": 0.45,
    },
    authentication_primary_probability=0.55,
    local_file_access_probability=0.65,
    connection_hotspot_fraction=0.22,
    service_locality_probability=0.35,
    attack_technique_weights={
        "credential_access": 0.10,
        "remote_authentication": 0.10,
        "lateral_connection": 0.20,
        "remote_execution": 0.20,
        "service_execution": 0.40,
    },
)


ENTERPRISE_C_PROFILE = EnterpriseProfile(
    enterprise_id="enterprise_c",
    profile_name="hybrid_enterprise",
    entity_counts={
        "User": 55,
        "Host": 55,
        "Server": 35,
        "DomainController": 4,
        "Process": 120,
        "File": 70,
        "Service": 55,
    },
    active_compute_nodes=55,
    admin_assignments_per_user=1,
    process_server_probability=0.48,
    file_server_probability=0.55,
    service_server_concentration=0.30,
    benign_relation_weights={
        "authenticates_to": 0.22,
        "connects_to": 0.25,
        "spawns": 0.13,
        "accesses": 0.18,
        "invokes": 0.22,
    },
    authentication_primary_probability=0.72,
    local_file_access_probability=0.72,
    connection_hotspot_fraction=0.15,
    service_locality_probability=0.55,
    attack_technique_weights={
        "credential_access": 0.20,
        "remote_authentication": 0.20,
        "lateral_connection": 0.25,
        "remote_execution": 0.20,
        "service_execution": 0.15,
    },
)


ENTERPRISE_D_PROFILE = EnterpriseProfile(
    enterprise_id="enterprise_d",
    profile_name="linux_infrastructure",
    entity_counts={
        "User": 18,
        "Host": 20,
        "Server": 80,
        "DomainController": 2,
        "Process": 180,
        "File": 110,
        "Service": 35,
    },
    active_compute_nodes=75,
    admin_assignments_per_user=2,
    process_server_probability=0.92,
    file_server_probability=0.88,
    service_server_concentration=0.12,
    benign_relation_weights={
        "authenticates_to": 0.12,
        "connects_to": 0.26,
        "spawns": 0.27,
        "accesses": 0.28,
        "invokes": 0.07,
    },
    authentication_primary_probability=0.60,
    local_file_access_probability=0.94,
    connection_hotspot_fraction=0.12,
    service_locality_probability=0.80,
    attack_technique_weights={
        "credential_access": 0.18,
        "remote_authentication": 0.27,
        "lateral_connection": 0.20,
        "remote_execution": 0.30,
        "service_execution": 0.05,
    },
)


ENTERPRISE_E_PROFILE = EnterpriseProfile(
    enterprise_id="enterprise_e",
    profile_name="iot_ot_edge",
    entity_counts={
        "User": 10,
        "Host": 140,
        "Server": 12,
        "DomainController": 2,
        "Process": 75,
        "File": 25,
        "Service": 45,
    },
    active_compute_nodes=90,
    admin_assignments_per_user=0,
    process_server_probability=0.10,
    file_server_probability=0.15,
    service_server_concentration=0.75,
    benign_relation_weights={
        "authenticates_to": 0.04,
        "connects_to": 0.58,
        "spawns": 0.05,
        "accesses": 0.08,
        "invokes": 0.25,
    },
    authentication_primary_probability=0.35,
    local_file_access_probability=0.55,
    connection_hotspot_fraction=0.03,
    service_locality_probability=0.25,
    attack_technique_weights={
        "credential_access": 0.08,
        "remote_authentication": 0.07,
        "lateral_connection": 0.45,
        "remote_execution": 0.15,
        "service_execution": 0.25,
    },
)


ENTERPRISE_PROFILES = {
    profile.enterprise_id: profile
    for profile in (
        ENTERPRISE_A_PROFILE,
        ENTERPRISE_B_PROFILE,
        ENTERPRISE_C_PROFILE,
        ENTERPRISE_D_PROFILE,
        ENTERPRISE_E_PROFILE,
    )
}


def get_enterprise_profile(
    enterprise_id: str,
) -> EnterpriseProfile:
    try:
        profile = ENTERPRISE_PROFILES[enterprise_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown enterprise profile: {enterprise_id}"
        ) from error

    profile.validate()

    return profile


def enterprise_profile_ids() -> Tuple[str, ...]:
    return tuple(sorted(ENTERPRISE_PROFILES))
