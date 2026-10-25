//! Minimal S3 client: PUT and HEAD of single objects, path-style, SigV4.
//!
//! Why not an S3 SDK: the sampler needs exactly two calls (upload a JPEG,
//! check the bucket exists), against one in-cluster endpoint. `aws-sdk-s3`
//! brings in the whole Smithy runtime and a TLS stack, roughly 200 crates and
//! minutes of build time, for two requests. The signing algorithm (AWS
//! Signature Version 4) is short, fully specified and testable against the
//! worked examples AWS publishes, which the tests below reproduce exactly.
//!
//! Path-style addressing (`http://host/bucket/key`) because the homelab's S3
//! is reached by its Service name; virtual-host style (`bucket.host`) would
//! need wildcard DNS that does not exist in the cluster.
//!
//! TLS is out of scope: the homelab endpoint is plain HTTP inside the cluster
//! network, and the HTTP client is built without a TLS stack. An `https://`
//! endpoint is refused at startup with a clear error rather than failing
//! obscurely on the first upload.

use std::fmt::Write as _;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// Characters that SigV4 leaves unencoded in a URI path (RFC 3986 unreserved).
fn is_unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~')
}

/// Percent-encode one path segment or query value the way SigV4 requires.
pub fn uri_encode(input: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        if is_unreserved(b) || (keep_slash && b == b'/') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac(key: &[u8], data: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

#[derive(Clone)]
pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
}

impl std::fmt::Debug for Credentials {
    // Never print the secret, even in debug logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("access_key", &self.access_key)
            .finish_non_exhaustive()
    }
}

/// A request ready to sign: method, path, query and headers, all lower-case
/// header names. Only what the sampler uses is modelled.
pub struct Request<'a> {
    pub method: &'a str,
    /// Already-encoded absolute path, e.g. `/datasets/frames/a.jpg`.
    pub path: &'a str,
    /// (name, value) pairs, not yet encoded.
    pub query: Vec<(&'a str, &'a str)>,
    /// (lower-case name, value) pairs. Must include `host`.
    pub headers: Vec<(String, String)>,
    pub payload_sha256: &'a str,
}

/// `YYYYMMDDTHHMMSSZ` for a Unix timestamp, without a date library.
pub fn amz_date(unix_seconds: i64) -> String {
    // Civil-from-days (Howard Hinnant's algorithm), valid for the proleptic
    // Gregorian calendar; enough for any timestamp a pod will see.
    let days = unix_seconds.div_euclid(86_400);
    let secs = unix_seconds.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// Compute the `Authorization` header value for a request (SigV4, header-based).
///
/// `amz_date` must also be sent as the `x-amz-date` header (it must already be
/// in `request.headers`).
pub fn authorization(
    request: &Request<'_>,
    credentials: &Credentials,
    region: &str,
    amz_date: &str,
) -> String {
    let date = &amz_date[..8];
    let mut headers: Vec<(String, String)> = request
        .headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    headers.sort();
    let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let mut query: Vec<(String, String)> = request
        .query
        .iter()
        .map(|(k, v)| (uri_encode(k, false), uri_encode(v, false)))
        .collect();
    query.sort();
    let canonical_query = query
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&");
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        request.method,
        request.path,
        canonical_query,
        canonical_headers,
        signed_headers,
        request.payload_sha256
    );
    let scope = format!("{date}/{region}/s3/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );
    let k_date = hmac(format!("AWS4{}", credentials.secret_key).as_bytes(), date);
    let k_region = hmac(&k_date, region);
    let k_service = hmac(&k_region, "s3");
    let k_signing = hmac(&k_service, "aws4_request");
    let signature = hex(&hmac(&k_signing, &string_to_sign));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        credentials.access_key
    )
}

/// Upload and existence checks against one endpoint and bucket.
#[derive(Clone)]
pub struct Client {
    endpoint: String, // scheme://host[:port], no trailing slash
    host: String,     // host[:port] as sent in the Host header
    region: String,
    credentials: Credentials,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(endpoint: &str, region: &str, credentials: Credentials) -> Result<Self> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let Some(rest) = endpoint.strip_prefix("http://") else {
            if endpoint.starts_with("https://") {
                bail!(
                    "S3 endpoint {endpoint} uses https, but this build has no TLS stack \
                     (the homelab endpoint is plain HTTP inside the cluster); see ingest/README.md"
                );
            }
            bail!("S3 endpoint must start with http://, got {endpoint:?}");
        };
        if rest.is_empty() || rest.contains('/') {
            bail!("S3 endpoint must be scheme://host[:port] with no path, got {endpoint:?}");
        }
        let config = ureq::Agent::config_builder()
            // Status codes are inspected below so the S3 error body can be
            // logged; ureq would otherwise turn them into a bare error.
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(30)))
            .build();
        Ok(Self {
            host: rest.to_string(),
            endpoint,
            region: region.to_string(),
            credentials,
            agent: config.into(),
        })
    }

    fn object_path(bucket: &str, key: &str) -> String {
        format!("/{}/{}", uri_encode(bucket, false), uri_encode(key, true))
    }

    fn signed_headers(
        &self,
        method: &str,
        path: &str,
        payload_sha256: &str,
        extra: &[(&str, &str)],
    ) -> Vec<(String, String)> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let date = amz_date(now);
        let mut headers = vec![
            ("host".to_string(), self.host.clone()),
            (
                "x-amz-content-sha256".to_string(),
                payload_sha256.to_string(),
            ),
            ("x-amz-date".to_string(), date.clone()),
        ];
        for (k, v) in extra {
            headers.push((k.to_ascii_lowercase(), v.to_string()));
        }
        let auth = authorization(
            &Request {
                method,
                path,
                query: vec![],
                headers: headers.clone(),
                payload_sha256,
            },
            &self.credentials,
            &self.region,
            &date,
        );
        headers.push(("authorization".to_string(), auth));
        // ureq sets Host itself from the URL.
        headers.retain(|(k, _)| k != "host");
        headers
    }

    /// PUT one object. Blocking: call from `spawn_blocking`.
    pub fn put_object(
        &self,
        bucket: &str,
        key: &str,
        body: &[u8],
        content_type: &str,
    ) -> Result<()> {
        let path = Self::object_path(bucket, key);
        let digest = sha256_hex(body);
        let headers = self.signed_headers("PUT", &path, &digest, &[("content-type", content_type)]);
        let mut request = self.agent.put(format!("{}{}", self.endpoint, path));
        for (k, v) in &headers {
            request = request.header(k.as_str(), v.as_str());
        }
        let mut response = request
            .send(body)
            .with_context(|| format!("PUT s3://{bucket}/{key}"))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let detail = response.body_mut().read_to_string().unwrap_or_default();
            bail!("PUT s3://{bucket}/{key}: HTTP {status}: {}", detail.trim());
        }
        Ok(())
    }

    /// HEAD the bucket: fails fast at startup if the endpoint, credentials or
    /// bucket are wrong, instead of on the first frame.
    pub fn check_bucket(&self, bucket: &str) -> Result<()> {
        let path = format!("/{}", uri_encode(bucket, false));
        let empty = sha256_hex(b"");
        let headers = self.signed_headers("HEAD", &path, &empty, &[]);
        let mut request = self.agent.head(format!("{}{}", self.endpoint, path));
        for (k, v) in &headers {
            request = request.header(k.as_str(), v.as_str());
        }
        let response = request
            .call()
            .with_context(|| format!("HEAD s3://{bucket}"))?;
        match response.status().as_u16() {
            200..=299 => Ok(()),
            403 => bail!(
                "S3 refused access to bucket {bucket:?} (HTTP 403): check the s3-credentials Secret"
            ),
            404 => bail!("S3 bucket {bucket:?} does not exist (HTTP 404)"),
            status => bail!("HEAD s3://{bucket}: HTTP {status}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Worked examples from the AWS documentation, "Signature Calculations for
    // the Authorization Header: Transferring Payload in a Single Chunk (AWS
    // Signature Version 4)". The expected signatures are the ones published
    // there; matching them proves the canonical request, string to sign and
    // key derivation are byte-for-byte right.
    const KEY: &str = "AKIAIOSFODNN7EXAMPLE";
    const SECRET: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

    fn creds() -> Credentials {
        Credentials {
            access_key: KEY.into(),
            secret_key: SECRET.into(),
        }
    }

    #[test]
    fn aws_example_get_object() {
        let empty = sha256_hex(b"");
        assert_eq!(
            empty,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let request = Request {
            method: "GET",
            path: "/test.txt",
            query: vec![],
            headers: vec![
                ("host".into(), "examplebucket.s3.amazonaws.com".into()),
                ("range".into(), "bytes=0-9".into()),
                ("x-amz-content-sha256".into(), empty.clone()),
                ("x-amz-date".into(), "20130524T000000Z".into()),
            ],
            payload_sha256: &empty,
        };
        let auth = authorization(&request, &creds(), "us-east-1", "20130524T000000Z");
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, \
             Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
    }

    #[test]
    fn aws_example_put_object() {
        let body = b"Welcome to Amazon S3.";
        let digest = sha256_hex(body);
        assert_eq!(
            digest,
            "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
        );
        let request = Request {
            method: "PUT",
            path: &format!("/{}", uri_encode("test$file.text", true)),
            query: vec![],
            headers: vec![
                ("Host".into(), "examplebucket.s3.amazonaws.com".into()),
                ("Date".into(), "Fri, 24 May 2013 00:00:00 GMT".into()),
                ("x-amz-date".into(), "20130524T000000Z".into()),
                ("x-amz-storage-class".into(), "REDUCED_REDUNDANCY".into()),
                ("x-amz-content-sha256".into(), digest.clone()),
            ],
            payload_sha256: &digest,
        };
        let auth = authorization(&request, &creds(), "us-east-1", "20130524T000000Z");
        assert!(
            auth.ends_with(
                "SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class, \
                 Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"
            ),
            "{auth}"
        );
    }

    #[test]
    fn aws_example_list_objects_query_string() {
        // GET Bucket (List Objects) example: query parameters are sorted and
        // encoded in the canonical request.
        let empty = sha256_hex(b"");
        let request = Request {
            method: "GET",
            path: "/",
            query: vec![("prefix", "J"), ("max-keys", "2")],
            headers: vec![
                ("host".into(), "examplebucket.s3.amazonaws.com".into()),
                ("x-amz-content-sha256".into(), empty.clone()),
                ("x-amz-date".into(), "20130524T000000Z".into()),
            ],
            payload_sha256: &empty,
        };
        let auth = authorization(&request, &creds(), "us-east-1", "20130524T000000Z");
        assert!(
            auth.ends_with(
                "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
            ),
            "{auth}"
        );
    }

    #[test]
    fn uri_encoding() {
        assert_eq!(
            uri_encode("frames/cam 1/a+b.jpg", true),
            "frames/cam%201/a%2Bb.jpg"
        );
        assert_eq!(uri_encode("a/b", false), "a%2Fb");
        assert_eq!(uri_encode("Az09-._~", false), "Az09-._~");
        assert_eq!(uri_encode("é", false), "%C3%A9");
    }

    #[test]
    fn amz_date_formats_unix_time() {
        assert_eq!(amz_date(0), "19700101T000000Z");
        // 2013-05-24T00:00:00Z, the date of the AWS examples.
        assert_eq!(amz_date(1_369_353_600), "20130524T000000Z");
        // A leap day.
        assert_eq!(amz_date(1_709_210_096), "20240229T123456Z");
    }

    #[test]
    fn https_endpoints_are_refused_with_an_explanation() {
        let err = Client::new("https://s3.example.com", "us-east-1", creds())
            .err()
            .expect("https must be refused");
        assert!(err.to_string().contains("no TLS"), "{err}");
        assert!(Client::new("s3.local:9000", "us-east-1", creds()).is_err());
        assert!(Client::new("http://h:9000/path", "us-east-1", creds()).is_err());
        assert!(
            Client::new(
                "http://s3.ml-storage.svc.cluster.local:9000/",
                "us-east-1",
                creds()
            )
            .is_ok()
        );
    }

    #[test]
    fn object_paths_are_path_style() {
        assert_eq!(
            Client::object_path("datasets", "frames/cam-1/000001.jpg"),
            "/datasets/frames/cam-1/000001.jpg"
        );
    }

    #[test]
    fn debug_output_hides_the_secret() {
        let text = format!("{:?}", creds());
        assert!(text.contains(KEY) && !text.contains(SECRET));
    }
}
