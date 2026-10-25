{{/* Image for the current GPU setting. */}}
{{- define "crosswalk.image" -}}
{{- if not (has .Values.gpu.vendor (list "amd" "nvidia")) -}}
{{- fail "gpu.vendor must be amd or nvidia" -}}
{{- end -}}
{{- $img := ternary .Values.image.gpu .Values.image.cpu .Values.gpu.enabled -}}
{{- if and .Values.gpu.enabled (has $img.tag (list "rocm-unset" "cuda-unset")) -}}
{{- fail "gpu.enabled=true needs image.gpu.tag: build the GPU image (scripts/build-rocm-image.sh, or docker/Dockerfile.cuda for NVIDIA) and set its tag" -}}
{{- end -}}
image: "{{ $img.repository }}:{{ $img.tag }}"
imagePullPolicy: {{ $img.pullPolicy }}
{{- end -}}

{{/* The CPU image, whatever gpu.enabled says: the streaming stages after
inference never need a GPU. */}}
{{- define "crosswalk.cpuImage" -}}
image: "{{ .Values.image.cpu.repository }}:{{ .Values.image.cpu.tag }}"
imagePullPolicy: {{ .Values.image.cpu.pullPolicy }}
{{- end -}}

{{/* Pod security for CPU-only pods: as crosswalk.podSecurity, without the
GPU groups and node placement. */}}
{{- define "crosswalk.cpuPodSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001
  seccompProfile:
    type: RuntimeDefault
{{- end -}}

{{/* Common labels. */}}
{{- define "crosswalk.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/part-of: crosswalk-world-cup
{{- end -}}

{{/*
Pod-level security. Non-root, no privilege escalation, RuntimeDefault seccomp.
With a GPU, the video and render groups are added so the non-root user can
open /dev/kfd and /dev/dri/renderD*. That is the one relaxation ROCm needs;
no privileged mode, no hostPath, no added
capabilities (the device plugin injects the device nodes).
*/}}
{{- define "crosswalk.podSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001
  {{- if and .Values.gpu.enabled (eq .Values.gpu.vendor "amd") }}
  supplementalGroups:
    - {{ .Values.gpu.videoGid }}
    - {{ .Values.gpu.renderGid }}
  {{- end }}
  seccompProfile:
    type: RuntimeDefault
{{- with .Values.gpu.nodeSelector }}
{{- if $.Values.gpu.enabled }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- with .Values.gpu.tolerations }}
{{- if $.Values.gpu.enabled }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "crosswalk.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
{{- end -}}

{{/* Resources: the GPU or CPU profile, plus the GPU limit when enabled. */}}
{{- define "crosswalk.resources" -}}
{{- $profile := ternary .resources.gpu .resources.cpu .root.Values.gpu.enabled -}}
{{- $limits := deepCopy ($profile.limits | default dict) -}}
{{- if .root.Values.gpu.enabled -}}
{{- $_ := set $limits .root.Values.gpu.resource (int (.gpus | default 1)) -}}
{{- end -}}
resources:
  requests:
    {{- toYaml ($profile.requests | default dict) | nindent 4 }}
  limits:
    {{- toYaml $limits | nindent 4 }}
{{- end -}}

{{/* S3 and MLflow environment shared by every container. */}}
{{- define "crosswalk.env" -}}
- name: AWS_ENDPOINT_URL_S3
  value: {{ .Values.s3.endpoint | quote }}
- name: AWS_DEFAULT_REGION
  value: {{ .Values.s3.region | quote }}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3.credentialsSecret }}
      key: AWS_ACCESS_KEY_ID
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3.credentialsSecret }}
      key: AWS_SECRET_ACCESS_KEY
- name: MLFLOW_TRACKING_URI
  value: {{ .Values.mlflow.trackingUri | quote }}
- name: KAFKA_BOOTSTRAP_SERVERS
  value: {{ .Values.kafka.bootstrapServers | quote }}
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
- name: HOME
  value: /tmp
{{- if and .Values.gpu.enabled (eq .Values.gpu.vendor "amd") .Values.gpu.hsaOverrideGfxVersion }}
- name: HSA_OVERRIDE_GFX_VERSION
  value: {{ .Values.gpu.hsaOverrideGfxVersion | quote }}
{{- end }}
{{- end -}}
