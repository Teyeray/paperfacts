"""What a reader is told about each column of a dataset: display text, in no cache key.

The labels and descriptions are the profile's display copy, so building them in :mod:`paperfacts.dataset`, whose
source is hashed into ``comparison_key``, would rename every stored comparison for an edit that changes no
verdict. They are not stored with a table either: the web library and the workbook build them from the profile
they run under, so an edited label shows without a re-run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from paperfacts.fields import Cardinality, FieldKind
from paperfacts.profile import DomainProfile


class FieldColumn(BaseModel):
    """What a reader needs to know about one column, built once for both the web UI and the Excel sheet.

    ``label`` and ``description`` are display only and may be empty when the profile declares neither;
    ``unit`` is absent for a text field. ``kind`` and ``cardinality`` decide how a cell of the column is written
    out (the workbook's :func:`paperfacts.workbook.format_cell`, the web's table.js and tsv.js), so a value is
    never formatted by its shape alone.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str = ""
    unit: str | None = None
    scope: str
    description: str = ""
    # A default only because dataset.json files written before the list was dropped from disk still carry one;
    # every list the server hands out is built by field_columns, which always sets it.
    kind: FieldKind = "text"
    cardinality: Cardinality = "one"
    # The entity type a sample-level column describes, in a profile that declares entity types; None for a
    # paper-level column and for every column of a profile without them (all of whose samples are one kind).
    entity: str | None = None


def field_columns(profile: DomainProfile) -> tuple[FieldColumn, ...]:
    """``profile``'s fields as columns, in the order the dataset writes them."""
    return tuple(
        FieldColumn(
            name=spec.name,
            label=spec.label,
            unit=spec.canonical_unit,
            scope="sample" if spec.is_sample_level else "paper",
            description=spec.description_zh,
            kind=spec.kind,
            cardinality=spec.cardinality,
            entity=spec.entity,
        )
        for spec in profile.fields
    )
