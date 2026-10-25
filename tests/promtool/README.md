Alert rule unit tests for `promtool test rules`. CI renders the chart's
PrometheusRule (pipeline enabled), extracts `spec` into `rules.yaml` next to
these files, and runs `promtool test rules alerts-test.yaml`.
