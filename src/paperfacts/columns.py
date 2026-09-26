"""What a reader is told about each column of a dataset: display text, in no cache key.

The labels and descriptions are the profile's display copy, so building them in :mod:`paperfacts.dataset`, whose
source is hashed into ``comparison_key``, would rename every stored comparison for an edit that changes no
verdict. They are not stored with a table either: the web library and the workbook build them from the profile
they run under, so an edited label shows without a re-run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from paperfacts.profile import DomainProfile


class FieldColumn(BaseModel):
    """What a reader needs to know about one column, built once for both the web UI and the Excel sheet.

    ``label`` and ``description`` are display only and may be empty when the profile declares neither;
    ``unit`` is absent for a text field.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str = ""
    unit: str | None = None
    scope: str
    description: str = ""


def field_columns(profile: DomainProfile) -> tuple[FieldColumn, ...]:
    """``profile``'s fields as columns, in the order the dataset writes them."""
    return tuple(
        FieldColumn(
            name=spec.name,
            label=spec.label,
            unit=spec.canonical_unit,
            scope="sample" if spec.is_sample_level else "paper",
            description=spec.description_zh,
        )
        for spec in profile.fields
    )
