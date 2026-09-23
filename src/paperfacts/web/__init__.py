"""Web interface: upload a PDF, run the full pipeline, and visualize the two-lane extraction and
alignment results together with their page+bbox provenance.

The backend is FastAPI (:mod:`app`); the frontend is build-step-free static files (``static/``),
served by the same process. All business logic still lives in :mod:`paperfacts.workflow`; this
package only handles the document library (:mod:`documents`), background jobs (:mod:`jobs`), and
the HTTP mapping.
"""
