"""Milestone 3: the streaming pipeline between the stages.

Kafka topics connect the stages; each stage is a consumer group that can be
scaled by adding pods (up to the topic's partition count)::

    crosswalk-ingest (Rust) ── crosswalk.frames.v1 ──► inference ── crosswalk.detections.v1
        ──► tracking ── crosswalk.tracks.v1 ──► indicators ── crosswalk.indicators.v1 ──► ranking

Everything here works against :class:`~crosswalk.stream.bus.MemoryBus` (in
process, used by tests and ``crosswalk demo``) and
:class:`~crosswalk.stream.bus.KafkaBus` (confluent-kafka) through the same
interface.
"""
