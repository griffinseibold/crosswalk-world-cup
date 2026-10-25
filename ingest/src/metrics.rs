//! Prometheus metrics and the tiny HTTP server that exposes them.
//!
//! Metric names share the `crosswalk_ingest_` prefix and are listed in
//! [`METRIC_NAMES`]; the Python test suite reads that list from this file and
//! checks the Grafana dashboard and alert rules only query names that exist.
//!
//! The server answers `/metrics`, `/healthz` (process alive) and `/readyz`
//! (Kafka and S3 reached at startup). It is a hand-rolled HTTP/1.0 responder
//! on a tokio listener: three read-only GET endpoints for probes and a
//! scraper do not justify a web framework and its dependency tree.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::watch;

/// Every exported family name (counters are exposed with `_total`,
/// histograms add `_bucket`/`_sum`/`_count`). Read by the Python tests from
/// this source file; used here only by the test below.
#[cfg_attr(not(test), allow(dead_code))]
pub const METRIC_NAMES: &[&str] = &[
    "crosswalk_ingest_frames_decoded_total",
    "crosswalk_ingest_frames_sampled_total",
    "crosswalk_ingest_frames_published_total",
    "crosswalk_ingest_frames_dropped_total",
    "crosswalk_ingest_errors_total",
    "crosswalk_ingest_buffer_depth",
    "crosswalk_ingest_buffer_capacity",
    "crosswalk_ingest_backpressure_seconds_total",
    "crosswalk_ingest_encode_seconds",
    "crosswalk_ingest_upload_seconds",
    "crosswalk_ingest_publish_seconds",
    "crosswalk_ingest_frame_age_seconds",
    "crosswalk_ingest_streams_completed_total",
    "crosswalk_ingest_ready",
];

pub type Labels = Vec<(&'static str, String)>;

pub fn camera(camera_id: &str) -> Labels {
    vec![("camera", camera_id.to_string())]
}

// Seconds, dense from 1 ms to 2 s: JPEG encoding and S3/Kafka round trips
// inside one cluster live there.
const LATENCY_BUCKETS: [f64; 14] = [
    0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0,
];

fn latency_histogram() -> Histogram {
    Histogram::new(LATENCY_BUCKETS)
}

#[derive(Clone)]
pub struct Metrics {
    pub decoded: Family<Labels, Counter>,
    pub sampled: Family<Labels, Counter>,
    pub published: Family<Labels, Counter>,
    /// Label `reason`: buffer_full (drop policy) or shutdown (decoded but
    /// still queued when the grace period ran out).
    pub dropped: Family<Labels, Counter>,
    /// Label `stage`: decode, encode, upload, publish, startup.
    pub errors: Family<Labels, Counter>,
    pub buffer_depth: Gauge,
    pub buffer_capacity: Gauge,
    /// Time producers spent blocked waiting for buffer room (block policy):
    /// the direct measure of backpressure. Stored as microseconds, exported
    /// as seconds by `rate()` users dividing by 1e6 is avoided by exporting
    /// a float counter.
    pub backpressure: Counter<f64, AtomicU64>,
    pub encode_seconds: Histogram,
    pub upload_seconds: Histogram,
    pub publish_seconds: Histogram,
    /// decoded_at -> Kafka acknowledged, per frame: what the sampler adds to
    /// end-to-end latency.
    pub frame_age_seconds: Histogram,
    pub streams_completed: Family<Labels, Counter>,
    pub ready: Gauge,
    registry: Arc<Registry>,
}

impl Metrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let m = Self {
            decoded: Family::default(),
            sampled: Family::default(),
            published: Family::default(),
            dropped: Family::default(),
            errors: Family::default(),
            buffer_depth: Gauge::default(),
            buffer_capacity: Gauge::default(),
            backpressure: Counter::default(),
            encode_seconds: latency_histogram(),
            upload_seconds: latency_histogram(),
            publish_seconds: latency_histogram(),
            frame_age_seconds: Histogram::new([
                0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0,
            ]),
            streams_completed: Family::default(),
            ready: Gauge::default(),
            registry: Arc::new(Registry::default()),
        };
        // prometheus-client appends _total to counters itself.
        registry.register(
            "crosswalk_ingest_frames_decoded",
            "Frames FFmpeg decoded",
            m.decoded.clone(),
        );
        registry.register(
            "crosswalk_ingest_frames_sampled",
            "Frames the sampling policy kept",
            m.sampled.clone(),
        );
        registry.register(
            "crosswalk_ingest_frames_published",
            "Frame events acknowledged by Kafka (image already in S3)",
            m.published.clone(),
        );
        registry.register(
            "crosswalk_ingest_frames_dropped",
            "Sampled frames discarded without publishing, by reason",
            m.dropped.clone(),
        );
        registry.register(
            "crosswalk_ingest_errors",
            "Failures by stage",
            m.errors.clone(),
        );
        registry.register(
            "crosswalk_ingest_buffer_depth",
            "Frames waiting between the decoder and the publishers",
            m.buffer_depth.clone(),
        );
        registry.register(
            "crosswalk_ingest_buffer_capacity",
            "Configured buffer size",
            m.buffer_capacity.clone(),
        );
        registry.register(
            "crosswalk_ingest_backpressure_seconds",
            "Time the decoder spent blocked waiting for buffer room",
            m.backpressure.clone(),
        );
        registry.register(
            "crosswalk_ingest_encode_seconds",
            "Time to JPEG-encode one sampled frame",
            m.encode_seconds.clone(),
        );
        registry.register(
            "crosswalk_ingest_upload_seconds",
            "Time to upload one frame to S3",
            m.upload_seconds.clone(),
        );
        registry.register(
            "crosswalk_ingest_publish_seconds",
            "Time for Kafka to acknowledge one frame event",
            m.publish_seconds.clone(),
        );
        registry.register(
            "crosswalk_ingest_frame_age_seconds",
            "Decode to Kafka acknowledgement, per published frame",
            m.frame_age_seconds.clone(),
        );
        registry.register(
            "crosswalk_ingest_streams_completed",
            "Passes over a clip finished (end event published)",
            m.streams_completed.clone(),
        );
        registry.register(
            "crosswalk_ingest_ready",
            "1 once Kafka and S3 were reached and topics exist",
            m.ready.clone(),
        );
        Self {
            registry: Arc::new(registry),
            ..m
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        encode(&mut out, &self.registry).expect("writing to a String cannot fail");
        out
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Liveness/readiness flags shared with the server.
#[derive(Default)]
pub struct Health {
    pub ready: AtomicBool,
    pub requests: AtomicU64,
}

fn response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.0 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Route one request line to a full HTTP response.
pub fn route(request_line: &str, metrics: &Metrics, health: &Health) -> String {
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("").split('?').next().unwrap_or("");
    if method != "GET" {
        return response("405 Method Not Allowed", "text/plain", "GET only\n");
    }
    match path {
        "/healthz" => response("200 OK", "text/plain", "ok\n"),
        "/readyz" if health.ready.load(Ordering::Relaxed) => {
            response("200 OK", "text/plain", "ready\n")
        }
        "/readyz" => response("503 Service Unavailable", "text/plain", "starting\n"),
        "/metrics" => response(
            "200 OK",
            "application/openmetrics-text; version=1.0.0; charset=utf-8",
            &metrics.render(),
        ),
        _ => response("404 Not Found", "text/plain", "not found\n"),
    }
}

/// Serve until `shutdown` flips to true.
pub async fn serve(
    listener: TcpListener,
    metrics: Metrics,
    health: Arc<Health>,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            _ = shutdown.changed() => return,
            accepted = listener.accept() => {
                let Ok((mut socket, _)) = accepted else { continue };
                let metrics = metrics.clone();
                let health = Arc::clone(&health);
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let read = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        socket.read(&mut buf),
                    )
                    .await;
                    let Ok(Ok(n)) = read else { return };
                    let text = String::from_utf8_lossy(&buf[..n]);
                    let line = text.lines().next().unwrap_or("");
                    health.requests.fetch_add(1, Ordering::Relaxed);
                    let reply = route(line, &metrics, &health);
                    let _ = socket.write_all(reply.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_name_is_exported() {
        let m = Metrics::new();
        m.decoded.get_or_create(&camera("c")).inc();
        m.sampled.get_or_create(&camera("c")).inc();
        m.published.get_or_create(&camera("c")).inc();
        m.dropped
            .get_or_create(&vec![
                ("camera", "c".into()),
                ("reason", "buffer_full".into()),
            ])
            .inc();
        m.errors
            .get_or_create(&vec![("stage", "upload".into())])
            .inc();
        m.streams_completed.get_or_create(&camera("c")).inc();
        m.backpressure.inc_by(0.25);
        m.encode_seconds.observe(0.004);
        let text = m.render();
        for name in METRIC_NAMES {
            assert!(text.contains(name), "{name} missing from:\n{text}");
        }
        assert!(text.contains("crosswalk_ingest_frames_published_total{camera=\"c\"} 1"));
        assert!(text.contains("crosswalk_ingest_backpressure_seconds_total 0.25"));
        assert!(text.ends_with("# EOF\n"));
    }

    #[test]
    fn routes() {
        let m = Metrics::new();
        let health = Health::default();
        assert!(route("GET /healthz HTTP/1.1", &m, &health).starts_with("HTTP/1.0 200"));
        assert!(route("GET /readyz HTTP/1.1", &m, &health).starts_with("HTTP/1.0 503"));
        health.ready.store(true, Ordering::Relaxed);
        assert!(route("GET /readyz HTTP/1.1", &m, &health).starts_with("HTTP/1.0 200"));
        let metrics = route("GET /metrics?x=1 HTTP/1.1", &m, &health);
        assert!(metrics.contains("crosswalk_ingest_ready"));
        assert!(route("GET /nope HTTP/1.1", &m, &health).starts_with("HTTP/1.0 404"));
        assert!(route("POST /metrics HTTP/1.1", &m, &health).starts_with("HTTP/1.0 405"));
    }

    #[tokio::test]
    async fn server_answers_over_tcp_and_stops() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = watch::channel(false);
        let health = Arc::new(Health::default());
        let task = tokio::spawn(serve(listener, Metrics::new(), Arc::clone(&health), rx));
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();
        let mut reply = String::new();
        stream.read_to_string(&mut reply).await.unwrap();
        assert!(
            reply.starts_with("HTTP/1.0 200") && reply.ends_with("ok\n"),
            "{reply}"
        );
        tx.send(true).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("server stopped")
            .unwrap();
    }
}
