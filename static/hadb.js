const BAND_COLOR = {
  "Low": "var(--low)",
  "Moderate": "var(--moderate)",
  "High": "var(--high)",
  "Very high": "var(--veryhigh)",
};

// A severe patient with a whole-domain deletion and no circulating antigen:
// the high-risk corner of the cohort, and a quick way to see the model move.
const DEMO = {
  effect: "Large Deletion",
  domain: "A2",
  exon: "14",
  aa_position: "",
  aa_first: "",
  aa_last: "",
  severity: "Severe",
  fviii_activity: "0.5",
  crm_type: "I",
  fviii_antigen: "",
  region: "europe_other",
};

function payload() {
  return Object.fromEntries(new FormData(document.getElementById("form")).entries());
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
    const up = r.contribution > 0;
    eff.className = up ? "up" : "down";
    eff.textContent = `${up ? "increases" : "lowers"} risk (${r.contribution.toFixed(4)})`;
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
    const res = await fetch("/api/hadb/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Prediction failed.");
      return;
    }
    document.getElementById("out").hidden = false;
    document.getElementById("prob").textContent = data.risk_percent;
    document.getElementById("band").textContent = data.band + " risk";
    document.getElementById("band").style.color = BAND_COLOR[data.band] || "";
    document.getElementById("call").textContent =
      `${data.call} — at the ${data.threshold_rule.replace(/_/g, " ")} threshold of ${data.threshold}`;
    const fill = document.getElementById("fill");
    fill.style.width = `${Math.min(100, data.risk * 100)}%`;
    fill.style.background = BAND_COLOR[data.band] || "";
    renderExplanation(data.explanation);
    document.getElementById("disclaimer").textContent = data.disclaimer || "";
    document.getElementById("out").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    alert(`Request failed: ${err}`);
  } finally {
    button.disabled = false;
    button.textContent = "Estimate risk";
  }
}

function loadDemo() {
  for (const [key, value] of Object.entries(DEMO)) {
    const el = document.getElementById(key);
    if (el) el.value = value;
  }
}

document.getElementById("form").addEventListener("submit", submit);
document.getElementById("demo").addEventListener("click", loadDemo);
