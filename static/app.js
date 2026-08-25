const BAND_COLOR = {
  "Low": "var(--low)",
  "Moderate": "var(--moderate)",
  "High": "var(--high)",
  "Very high": "var(--veryhigh)",
};

const DEMO = {
  hgvs_cdna: "c.6496C>T",
  hgvs_protein: "p.(Arg2166*)",
  variant_type: "Nonsense",
  mechanism: "Substitution",
  exon: "23",
  domain: "C1",
  subtype: "Light chain",
  in_poly_a: "N",
  severity: "Severe",
};

function payload() {
  const form = document.getElementById("form");
  return Object.fromEntries(new FormData(form).entries());
}

function renderExplanation(rows) {
  const body = document.querySelector("#expl tbody");
  body.innerHTML = "";
  if (!rows || !rows.length) {
    body.innerHTML = '<tr><td colspan="2">No attribution available.</td></tr>';
    return;
  }
  for (const r of rows) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = r.feature.replace(/_/g, " ");
    const eff = document.createElement("td");
    const up = r.shap > 0;
    eff.className = up ? "up" : "down";
    eff.textContent = `${up ? "increases" : "decreases"} risk (${r.shap.toFixed(3)})`;
    tr.append(name, eff);
    body.append(tr);
  }
}

async function submit(event) {
  event.preventDefault();
  const button = document.getElementById("go");
  button.disabled = true;
  button.textContent = "Estimating...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    const pct = (data.probability * 100).toFixed(1);
    const colour = BAND_COLOR[data.risk_band] || "var(--primary)";
    document.getElementById("prob").textContent = `${pct}%`;
    document.getElementById("fill").style.width = `${Math.min(data.probability * 100, 100)}%`;
    document.getElementById("fill").style.background = colour;
    const band = document.getElementById("band");
    band.textContent = `${data.risk_band} risk`;
    band.style.background = colour;
    document.getElementById("call").textContent =
      `${data.prediction} at threshold ${data.threshold.toFixed(3)} (${data.threshold_rule.replace(/_/g, " ")})`;
    renderExplanation(data.explanation);
    document.getElementById("disclaimer").textContent = data.disclaimer || "";
    document.getElementById("out").hidden = false;
    document.getElementById("out").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    document.getElementById("out").hidden = false;
    document.getElementById("prob").textContent = "—";
    document.getElementById("call").textContent = String(err.message || err);
    renderExplanation([]);
  } finally {
    button.disabled = false;
    button.textContent = "Estimate risk";
  }
}

document.getElementById("form").addEventListener("submit", submit);
document.getElementById("demo").addEventListener("click", () => {
  for (const [k, v] of Object.entries(DEMO)) {
    const el = document.getElementById(k);
    if (el) el.value = v;
  }
});
