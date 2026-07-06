from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import random

from fedrelshield.data.attacks import AttackCampaign
from fedrelshield.data.benign import SecurityEvent
from fedrelshield.data.topology import EnterpriseTopology


Triple = Tuple[str, str, str]


@dataclass(frozen=True)
class CampaignSplit:
    train_campaign_ids: Tuple[str, ...]
    valid_campaign_ids: Tuple[str, ...]
    test_campaign_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ExportStatistics:
    topology_unique_triples: int
    benign_unique_triples: int

    train_attack_raw_events: int
    valid_attack_raw_events: int
    test_attack_raw_events: int

    train_attack_unique_triples: int
    valid_attack_unique_triples: int
    test_attack_unique_triples: int

    valid_removed_by_topology_overlap: int
    valid_removed_by_benign_overlap: int
    valid_removed_by_train_attack_overlap: int

    test_removed_by_topology_overlap: int
    test_removed_by_benign_overlap: int
    test_removed_by_train_attack_overlap: int

    test_removed_by_valid_overlap: int

    final_train_triples: int
    final_valid_triples: int
    final_test_triples: int

@dataclass(frozen=True)
class AttackOccurrence:
    campaign_id: str
    step_index: int
    event_id: str
    technique: str
    campaign_split: str

@dataclass(frozen=True)
class ExportedAttackStep:
    step_index: int
    technique: str
    event_id: str
    timestamp: int
    triple: Triple


@dataclass(frozen=True)
class ExportedCampaign:
    campaign_id: str
    campaign_split: str
    initial_timestamp: int
    steps: Tuple[ExportedAttackStep, ...]

@dataclass(frozen=True)
class TripleProvenance:
    triple: Triple
    is_topology_fact: bool
    benign_event_ids: Tuple[str, ...]
    attack_occurrences: Tuple[AttackOccurrence, ...]
    in_training_graph: bool
    link_prediction_split: str
    retained_as_link_prediction_target: bool


@dataclass(frozen=True)
class ExportedDataset:
    train_triples: Tuple[Triple, ...]
    valid_triples: Tuple[Triple, ...]
    test_triples: Tuple[Triple, ...]
    campaign_split: CampaignSplit
    statistics: ExportStatistics
    provenance: Tuple[TripleProvenance, ...]
    campaigns: Tuple[ExportedCampaign, ...]


class KGExporter:
    def __init__(
        self,
        seed: int = 1024,
        train_campaigns: int = 8,
        valid_campaigns: int = 2,
        test_campaigns: int = 2,
    ):
        if min(
            train_campaigns,
            valid_campaigns,
            test_campaigns,
        ) < 0:
            raise ValueError(
                "campaign split counts must be non-negative"
            )

        self.seed = seed
        self.train_campaigns = train_campaigns
        self.valid_campaigns = valid_campaigns
        self.test_campaigns = test_campaigns

    def build(
        self,
        topology: EnterpriseTopology,
        benign_events: List[SecurityEvent],
        attack_campaigns: List[AttackCampaign],
    ) -> ExportedDataset:
        campaign_split = self._split_campaigns(
            attack_campaigns,
        )

        exported_campaigns = self._build_exported_campaigns(
            attack_campaigns=attack_campaigns,
            campaign_split=campaign_split,
        )

        topology_triples = {
            (
                edge.head,
                edge.relation,
                edge.tail,
            )
            for edge in topology.edges
        }

        benign_triples = {
            (
                event.head,
                event.relation,
                event.tail,
            )
            for event in benign_events
        }

        campaigns_by_id = {
            campaign.campaign_id: campaign
            for campaign in attack_campaigns
        }

        # ---------------------------------------------------------
        # Raw attack-event counts
        # ---------------------------------------------------------

        train_attack_raw_events = self._campaign_event_count(
            campaign_split.train_campaign_ids,
            campaigns_by_id,
        )

        valid_attack_raw_events = self._campaign_event_count(
            campaign_split.valid_campaign_ids,
            campaigns_by_id,
        )

        test_attack_raw_events = self._campaign_event_count(
            campaign_split.test_campaign_ids,
            campaigns_by_id,
        )

        # ---------------------------------------------------------
        # Unique attack triples
        # ---------------------------------------------------------

        train_attack_triples = self._campaign_triples(
            campaign_split.train_campaign_ids,
            campaigns_by_id,
        )

        valid_attack_triples = self._campaign_triples(
            campaign_split.valid_campaign_ids,
            campaigns_by_id,
        )

        test_attack_triples = self._campaign_triples(
            campaign_split.test_campaign_ids,
            campaigns_by_id,
        )

        # ---------------------------------------------------------
        # Training graph
        # ---------------------------------------------------------

        train_triples = (
            topology_triples
            | benign_triples
            | train_attack_triples
        )

        # ---------------------------------------------------------
        # Validation overlap diagnostics
        # ---------------------------------------------------------

        valid_overlap_with_topology = (
            valid_attack_triples
            & topology_triples
        )

        valid_overlap_with_benign = (
            valid_attack_triples
            & benign_triples
        )

        valid_overlap_with_train_attack = (
            valid_attack_triples
            & train_attack_triples
        )

        # Validation link-prediction targets must not already occur
        # in the training graph.
        valid_triples = (
            valid_attack_triples
            - train_triples
        )

        # ---------------------------------------------------------
        # Test overlap diagnostics
        # ---------------------------------------------------------

        test_overlap_with_topology = (
            test_attack_triples
            & topology_triples
        )

        test_overlap_with_benign = (
            test_attack_triples
            & benign_triples
        )

        test_overlap_with_train_attack = (
            test_attack_triples
            & train_attack_triples
        )

        # Test link-prediction targets must not already occur in
        # the training graph.
        test_after_train_filter = (
            test_attack_triples
            - train_triples
        )

        # Test targets must also not duplicate surviving validation
        # targets.
        test_overlap_with_valid = (
            test_after_train_filter
            & valid_triples
        )

        test_triples = (
            test_after_train_filter
            - valid_triples
        )

        provenance = self._build_provenance(
            topology_triples=topology_triples,
            benign_events=benign_events,
            attack_campaigns=attack_campaigns,
            campaign_split=campaign_split,
            train_triples=train_triples,
            valid_triples=valid_triples,
            test_triples=test_triples,
        )

        # ---------------------------------------------------------
        # Relation coverage validation
        # ---------------------------------------------------------

        self._validate_relation_coverage(
            train_triples,
            valid_triples,
            test_triples,
        )

        # ---------------------------------------------------------
        # Export diagnostics
        # ---------------------------------------------------------

        statistics = ExportStatistics(
            topology_unique_triples=len(
                topology_triples
            ),
            benign_unique_triples=len(
                benign_triples
            ),

            train_attack_raw_events=train_attack_raw_events,
            valid_attack_raw_events=valid_attack_raw_events,
            test_attack_raw_events=test_attack_raw_events,

            train_attack_unique_triples=len(
                train_attack_triples
            ),
            valid_attack_unique_triples=len(
                valid_attack_triples
            ),
            test_attack_unique_triples=len(
                test_attack_triples
            ),

            valid_removed_by_topology_overlap=len(
                valid_overlap_with_topology
            ),
            valid_removed_by_benign_overlap=len(
                valid_overlap_with_benign
            ),
            valid_removed_by_train_attack_overlap=len(
                valid_overlap_with_train_attack
            ),

            test_removed_by_topology_overlap=len(
                test_overlap_with_topology
            ),
            test_removed_by_benign_overlap=len(
                test_overlap_with_benign
            ),
            test_removed_by_train_attack_overlap=len(
                test_overlap_with_train_attack
            ),

            test_removed_by_valid_overlap=len(
                test_overlap_with_valid
            ),

            final_train_triples=len(
                train_triples
            ),
            final_valid_triples=len(
                valid_triples
            ),
            final_test_triples=len(
                test_triples
            ),
        )

        return ExportedDataset(
            train_triples=tuple(
                sorted(train_triples)
            ),
            valid_triples=tuple(
                sorted(valid_triples)
            ),
            test_triples=tuple(
                sorted(test_triples)
            ),
            campaign_split=campaign_split,
            statistics=statistics,
            provenance=provenance,
            campaigns=exported_campaigns,
        )

    def _split_campaigns(
        self,
        attack_campaigns: List[AttackCampaign],
    ) -> CampaignSplit:
        expected_campaigns = (
            self.train_campaigns
            + self.valid_campaigns
            + self.test_campaigns
        )

        if len(attack_campaigns) != expected_campaigns:
            raise ValueError(
                f"Expected {expected_campaigns} campaigns, "
                f"received {len(attack_campaigns)}"
            )

        campaign_ids = [
            campaign.campaign_id
            for campaign in attack_campaigns
        ]

        if len(campaign_ids) != len(set(campaign_ids)):
            raise ValueError(
                "Duplicate campaign IDs"
            )

        rng = random.Random(self.seed)

        shuffled_ids = list(campaign_ids)
        rng.shuffle(shuffled_ids)

        train_end = self.train_campaigns

        valid_end = (
            train_end
            + self.valid_campaigns
        )

        return CampaignSplit(
            train_campaign_ids=tuple(
                sorted(
                    shuffled_ids[:train_end]
                )
            ),
            valid_campaign_ids=tuple(
                sorted(
                    shuffled_ids[
                        train_end:valid_end
                    ]
                )
            ),
            test_campaign_ids=tuple(
                sorted(
                    shuffled_ids[valid_end:]
                )
            ),
        )

    def _campaign_triples(
        self,
        campaign_ids: Tuple[str, ...],
        campaigns_by_id: Dict[str, AttackCampaign],
    ) -> Set[Triple]:
        return {
            (
                event.head,
                event.relation,
                event.tail,
            )
            for campaign_id in campaign_ids
            for event in campaigns_by_id[
                campaign_id
            ].events
        }

    def _campaign_event_count(
        self,
        campaign_ids: Tuple[str, ...],
        campaigns_by_id: Dict[str, AttackCampaign],
    ) -> int:
        return sum(
            len(
                campaigns_by_id[
                    campaign_id
                ].events
            )
            for campaign_id in campaign_ids
        )

    def _validate_relation_coverage(
        self,
        train_triples: Set[Triple],
        valid_triples: Set[Triple],
        test_triples: Set[Triple],
    ):
        train_relations = {
            relation
            for _, relation, _ in train_triples
        }

        evaluation_relations = {
            relation
            for _, relation, _ in (
                valid_triples
                | test_triples
            )
        }

        missing_relations = (
            evaluation_relations
            - train_relations
        )

        if missing_relations:
            raise ValueError(
                "Evaluation relations missing "
                "from training graph: "
                f"{sorted(missing_relations)}"
            )

    def _build_provenance(
        self,
        topology_triples: Set[Triple],
        benign_events: List[SecurityEvent],
        attack_campaigns: List[AttackCampaign],
        campaign_split: CampaignSplit,
        train_triples: Set[Triple],
        valid_triples: Set[Triple],
        test_triples: Set[Triple],
    ) -> Tuple[TripleProvenance, ...]:
        benign_events_by_triple = {}
    
        for event in benign_events:
            triple = (
                event.head,
                event.relation,
                event.tail,
            )
    
            benign_events_by_triple.setdefault(
                triple,
                [],
            ).append(event.event_id)
    
        campaign_split_by_id = self._campaign_split_by_id(
            campaign_split
        )
    
        attack_occurrences_by_triple = {}
    
        for campaign in attack_campaigns:
            split_name = campaign_split_by_id[
                campaign.campaign_id
            ]
    
            for step in campaign.steps:
                event = step.event
    
                triple = (
                    event.head,
                    event.relation,
                    event.tail,
                )
    
                occurrence = AttackOccurrence(
                    campaign_id=campaign.campaign_id,
                    step_index=step.step_index,
                    event_id=event.event_id,
                    technique=step.technique,
                    campaign_split=split_name,
                )
    
                attack_occurrences_by_triple.setdefault(
                    triple,
                    [],
                ).append(occurrence)
    
        all_triples = (
            topology_triples
            | set(benign_events_by_triple)
            | set(attack_occurrences_by_triple)
        )
    
        provenance_records = []
    
        for triple in sorted(all_triples):
            if triple in valid_triples:
                link_prediction_split = "valid"
            elif triple in test_triples:
                link_prediction_split = "test"
            elif triple in train_triples:
                link_prediction_split = "train"
            else:
                link_prediction_split = "none"
    
            retained_as_target = (
                triple in valid_triples
                or triple in test_triples
            )
    
            benign_event_ids = tuple(
                sorted(
                    benign_events_by_triple.get(
                        triple,
                        [],
                    )
                )
            )
    
            attack_occurrences = tuple(
                sorted(
                    attack_occurrences_by_triple.get(
                        triple,
                        [],
                    ),
                    key=self._attack_occurrence_sort_key,
                )
            )
    
            provenance_records.append(
                TripleProvenance(
                    triple=triple,
                    is_topology_fact=(
                        triple in topology_triples
                    ),
                    benign_event_ids=benign_event_ids,
                    attack_occurrences=attack_occurrences,
                    in_training_graph=(
                        triple in train_triples
                    ),
                    link_prediction_split=link_prediction_split,
                    retained_as_link_prediction_target=retained_as_target,
                )
            )
    
        return tuple(provenance_records)
    
    def _campaign_split_by_id(
        self,
        campaign_split: CampaignSplit,
    ) -> Dict[str, str]:
        split_by_id = {}
    
        for campaign_id in campaign_split.train_campaign_ids:
            split_by_id[campaign_id] = "train"
    
        for campaign_id in campaign_split.valid_campaign_ids:
            split_by_id[campaign_id] = "valid"
    
        for campaign_id in campaign_split.test_campaign_ids:
            split_by_id[campaign_id] = "test"
    
        return split_by_id
    
    def _attack_occurrence_sort_key(
        self,
        occurrence: AttackOccurrence,
    ):
        return (
            occurrence.campaign_id,
            occurrence.step_index,
            occurrence.event_id,
        )

    def _build_exported_campaigns(
        self,
        attack_campaigns: List[AttackCampaign],
        campaign_split: CampaignSplit,
    ) -> Tuple[ExportedCampaign, ...]:
        split_by_id = self._campaign_split_by_id(
            campaign_split
        )
    
        exported_campaigns = []
    
        for campaign in attack_campaigns:
            if campaign.campaign_id not in split_by_id:
                raise ValueError(
                    "Attack campaign missing from campaign split: "
                    f"{campaign.campaign_id}"
                )
    
            ordered_steps = sorted(
                campaign.steps,
                key=lambda step: step.step_index,
            )
    
            if [
                step.step_index
                for step in ordered_steps
            ] != list(range(len(ordered_steps))):
                raise ValueError(
                    "Campaign steps must have contiguous indices "
                    f"starting at zero: {campaign.campaign_id}"
                )
    
            exported_steps = tuple(
                ExportedAttackStep(
                    step_index=step.step_index,
                    technique=step.technique,
                    event_id=step.event.event_id,
                    timestamp=step.event.timestamp,
                    triple=(
                        step.event.head,
                        step.event.relation,
                        step.event.tail,
                    ),
                )
                for step in ordered_steps
            )
    
            exported_campaigns.append(
                ExportedCampaign(
                    campaign_id=campaign.campaign_id,
                    campaign_split=split_by_id[
                        campaign.campaign_id
                    ],
                    initial_timestamp=campaign.initial_timestamp,
                    steps=exported_steps,
                )
            )
    
        return tuple(
            sorted(
                exported_campaigns,
                key=lambda campaign: campaign.campaign_id,
            )
        )

