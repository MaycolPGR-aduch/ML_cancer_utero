// Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Cliente Web

const $ = (id) => document.getElementById(id);

// --- Tabs ---
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// --- Health check ---
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const pills = [];
    pills.push(
      data.cnn_ready
        ? `<span class="pill pill-ok">CNN MobileNetV2: Listo</span>`
        : `<span class="pill pill-warn">CNN: Entrenando / No cargado</span>`
    );
    pills.push(
      data.tabular_ready
        ? `<span class="pill pill-ok">Módulo Tabular: Listo</span>`
        : `<span class="pill pill-warn">Tabular: No cargado</span>`
    );
    $("statusPills").innerHTML = pills.join("");
  } catch (err) {
    $("statusPills").innerHTML = `<span class="pill pill-warn">API desconectada</span>`;
  }
}

// --- CNN Upload & Samples ---
const dropzone = $("dropzone");
const imageInput = $("imageInput");

dropzone.addEventListener("click", () => imageInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    sendImageFile(e.dataTransfer.files[0]);
  }
});
imageInput.addEventListener("change", () => {
  if (imageInput.files.length > 0) {
    sendImageFile(imageInput.files[0]);
  }
});

async function sendImageFile(file) {
  $("cnnError").classList.add("hidden");
  $("cnnPlaceholder").textContent = "Analizando imagen citológica y generando mapa Grad-CAM...";
  $("cnnPlaceholder").classList.remove("hidden");
  $("cnnResults").classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch("/api/predict/image", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error en la predicción");
    renderCnnResult(data);
  } catch (err) {
    $("cnnError").textContent = err.message;
    $("cnnError").classList.remove("hidden");
    $("cnnPlaceholder").textContent = "Ocurrió un error al procesar la imagen.";
  }
}

async function runSampleImage(filename) {
  $("cnnError").classList.add("hidden");
  $("cnnPlaceholder").textContent = `Evaluando muestra ${filename} y generando Grad-CAM...`;
  $("cnnPlaceholder").classList.remove("hidden");
  $("cnnResults").classList.add("hidden");

  try {
    const res = await fetch(`/api/predict/sample/${encodeURIComponent(filename)}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error al evaluar muestra");
    renderCnnResult(data);
  } catch (err) {
    $("cnnError").textContent = err.message;
    $("cnnError").classList.remove("hidden");
  }
}

function renderCnnResult(data) {
  $("cnnPlaceholder").classList.add("hidden");
  $("cnnResults").classList.remove("hidden");

  const p = data.prediccion;
  const banner = $("cnnTriageBanner");
  banner.className = `triage-banner level-${p.nivel_tamizaje}`;
  $("cnnPredTitle").textContent = `${p.nombre_clinico} (${(p.confianza * 100).toFixed(1)}%)`;
  $("cnnPredDesc").textContent = p.recomendacion_apoyo;
  $("cnnAbnormalProb").textContent = `${(p.probabilidad_anormal_agregada * 100).toFixed(1)}%`;

  $("cnnMetaRow").innerHTML = `
    <span class="meta-chip">Archivo: ${data.entrada.archivo}</span>
    <span class="meta-chip">Original: ${data.entrada.tamano_original[0]}×${data.entrada.tamano_original[1]} → 224×224</span>
    <span class="meta-chip">Inferencia CNN: ${data.tiempos_ms.inferencia_cnn} ms</span>
    <span class="meta-chip">Grad-CAM: ${data.tiempos_ms.gradcam_xai} ms</span>
    <span class="meta-chip">Versión: ${data.version}</span>
  `;

  $("cnnPreviewImg").src = data.imagenes.entrada_224;
  $("cnnGradcamImg").src = data.imagenes.gradcam_overlay;

  $("cnnProbBars").innerHTML = p.probabilidades
    .map((item) => {
      const pct = (item.probabilidad * 100).toFixed(2);
      const clsType =
        item.categoria === "Anormal"
          ? "abnormal"
          : item.categoria === "Benigna"
          ? "benign"
          : "normal";
      return `
        <div class="prob-item">
          <div class="prob-header">
            <span><strong>${item.nombre_es}</strong> <span class="muted">(${item.clase})</span></span>
            <strong>${pct}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${clsType}" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    })
    .join("");

  const cleanJson = { ...data, imagenes: { entrada_224: "[base64 PNG omitido]", gradcam_overlay: "[base64 PNG omitido]" } };
  $("cnnRawJson").textContent = JSON.stringify(cleanJson, null, 2);
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    if (!data.samples || data.samples.length === 0) {
      $("sampleButtons").innerHTML = `<span class="muted">Aún no hay muestras exportadas (ejecuta train_eval_cnn.py).</span>`;
      return;
    }
    $("sampleButtons").innerHTML = data.samples
      .map(
        (s) =>
          `<button type="button" class="sample-btn" data-file="${s.filename}">${s.expected_class.replace("im_", "")}</button>`
      )
      .join("");
    document.querySelectorAll(".sample-btn").forEach((btn) => {
      btn.addEventListener("click", () => runSampleImage(btn.dataset.file));
    });
  } catch (err) {
    $("sampleButtons").innerHTML = `<span class="muted">No se pudieron cargar muestras.</span>`;
  }
}

// --- Tabular Form ---
let tabularSchemaCache = null;

async function loadTabularSchema() {
  try {
    const res = await fetch("/api/tabular/schema");
    if (!res.ok) return;
    tabularSchemaCache = await res.json();
    const container = $("tabularFields");
    container.innerHTML = tabularSchemaCache.features
      .map((f) => {
        if (f.type === "binary") {
          return `
            <div class="field-group">
              <label for="f_${f.name}">${f.label}</label>
              <select id="f_${f.name}" data-feature="${f.name}">
                <option value="">(Desconocido / Imputar)</option>
                <option value="0" ${f.default === 0 ? "selected" : ""}>No (0)</option>
                <option value="1" ${f.default === 1 ? "selected" : ""}>Sí (1)</option>
              </select>
            </div>
          `;
        }
        return `
          <div class="field-group">
            <label for="f_${f.name}">${f.label}</label>
            <input type="number" id="f_${f.name}" data-feature="${f.name}"
              min="${f.min ?? 0}" max="${f.max ?? 100}" step="${f.step ?? 1}"
              value="${f.default ?? f.median ?? 0}" placeholder="Mediana: ${f.median}" />
          </div>
        `;
      })
      .join("");
  } catch (err) {
    console.error(err);
  }
}

function applyTabularPreset(mode) {
  if (!tabularSchemaCache) return;
  const highValues = {
    Age: 42,
    "Number of sexual partners": 6,
    "First sexual intercourse": 15,
    "Num of pregnancies": 4,
    Smokes: 1,
    "Smokes (years)": 15,
    "Smokes (packs/year)": 8,
    "Hormonal Contraceptives": 1,
    "Hormonal Contraceptives (years)": 9,
    IUD: 0,
    "IUD (years)": 0,
    STDs: 1,
    "STDs (number)": 2,
    "STDs:condylomatosis": 1,
    "STDs:HPV": 1,
    "STDs: Number of diagnosis": 2,
  };
  document.querySelectorAll("[data-feature]").forEach((el) => {
    const name = el.dataset.feature;
    if (mode === "empty") {
      el.value = "";
    } else if (mode === "high") {
      el.value = name in highValues ? String(highValues[name]) : "0";
    } else {
      const meta = tabularSchemaCache.features.find((f) => f.name === name);
      el.value = meta ? String(meta.default ?? 0) : "0";
    }
  });
}

$("btnPresetLow").addEventListener("click", () => applyTabularPreset("low"));
$("btnPresetHigh").addEventListener("click", () => applyTabularPreset("high"));
$("btnPresetEmpty").addEventListener("click", () => applyTabularPreset("empty"));

$("tabularForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const values = {};
  document.querySelectorAll("[data-feature]").forEach((el) => {
    values[el.dataset.feature] = el.value === "" ? null : Number(el.value);
  });
  const res = await fetch("/api/predict/tabular", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "Error en inferencia tabular");
    return;
  }
  renderTabularResult(data);
});

function renderTabularResult(data) {
  $("tabPlaceholder").classList.add("hidden");
  $("tabResults").classList.remove("hidden");

  const p = data.prediccion;
  const banner = $("tabTriageBanner");
  banner.className = `triage-banner level-${p.nivel_tamizaje}`;
  $("tabRiskTitle").textContent = `Riesgo ${p.nivel_tamizaje} ${p.alerta_tamizaje ? "(Alerta Tamizaje)" : "(Bajo Umbral)"}`;
  $("tabRiskDesc").textContent = p.recomendacion_apoyo;
  $("tabProbVal").textContent = `${(p.probabilidad_riesgo_biopsia * 100).toFixed(1)}%`;

  $("tabMetaRow").innerHTML = `
    <span class="meta-chip">Modelo: ${data.modelo}</span>
    <span class="meta-chip">Latencia: ${data.latencia_inferencia_ms} ms</span>
    <span class="meta-chip">Variables imputadas: ${data.variables_imputadas}</span>
    <span class="meta-chip">Versión: ${data.version}</span>
  `;

  const probPct = Math.min(100, Math.max(0, p.probabilidad_riesgo_biopsia * 100));
  const thrPct = Math.min(100, Math.max(0, p.umbral_tamizaje * 100));
  $("tabThrText").textContent = `Prob: ${probPct.toFixed(1)}% | Umbral: ${thrPct.toFixed(1)}%`;
  $("tabProbBar").style.width = `${probPct}%`;
  $("tabThrMarker").style.left = `calc(${thrPct}% - 1.5px)`;

  $("tabTopFeatures").innerHTML = data.top_features
    .map((f) => {
      const pct = (f.importancia_relativa * 100).toFixed(1);
      return `
        <div class="prob-item">
          <div class="prob-header">
            <span>${f.variable}</span>
            <strong>${pct}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${Math.min(100, pct * 3)}%"></div>
          </div>
        </div>
      `;
    })
    .join("");

  $("tabRawJson").textContent = JSON.stringify(data, null, 2);
}

// --- TRL 3 Evidence Panel ---
async function loadEvidence() {
  try {
    const res = await fetch("/api/evidence");
    const ev = await res.json();

    const kpis = [];
    const cnn = ev.modulo_cnn;
    const tab = ev.modulo_tabular;

    if (cnn) {
      const m = cnn.metricas_test;
      const bin = m.tamizaje_clinico_binario;
      kpis.push(
        kpiCard("CNN Macro-F1 (Test)", (m.macro_f1 * 100).toFixed(2) + "%", `Accuracy: ${(m.accuracy * 100).toFixed(2)}%`),
        kpiCard("CNN Macro ROC-AUC", m.macro_roc_auc_ovr.toFixed(4), `PR-AUC: ${m.macro_pr_auc_ovr.toFixed(4)}`),
        kpiCard("Sensibilidad Anormal", (bin.sensibilidad_recall * 100).toFixed(2) + "%", `Especificidad: ${(bin.especificidad * 100).toFixed(2)}%`),
        kpiCard("Latencia CNN (CPU)", cnn.prueba_inferencia.latencia_promedio_ms_30_runs + " ms", "0 láminas compartidas (no leakage)")
      );

      $("cnnEvidenceSummary").innerHTML = [
        evRow("Versión / Arquitectura", `${cnn.identificacion.version} — ${cnn.entrenamiento.arquitectura}`),
        evRow("Dataset", `SIPaKMeD (${cnn.datos.total_imagenes} imágenes en 5 clases)`),
        evRow("Split sin fuga (láminas)", `Train ${cnn.datos.particion_sin_fuga.train.imagenes} | Val ${cnn.datos.particion_sin_fuga.val.imagenes} | Test ${cnn.datos.particion_sin_fuga.test.imagenes}`),
        evRow("Láminas compartidas", `${cnn.datos.particion_sin_fuga.laminas_compartidas_entre_splits} (verificado)`),
        evRow("Macro Recall / Precision", `${cnn.metricas_test.macro_recall} / ${cnn.metricas_test.macro_precision}`),
        evRow("Tiempo entrenamiento", `${cnn.entrenamiento.tiempo_entrenamiento_segundos} s (${cnn.entrenamiento.epocas_totales} épocas)`),
      ].join("");
    } else {
      $("cnnEvidenceSummary").innerHTML = `<p class="muted">Pendiente de completar entrenamiento CNN...</p>`;
    }

    if (tab) {
      const tm = tab.metricas_modelo_seleccionado.cv_5fold;
      const toof = tab.metricas_modelo_seleccionado.oof_con_umbral_tamizaje;
      kpis.push(
        kpiCard("Tabular ROC-AUC (5-Fold)", tm.roc_auc.toFixed(4), `PR-AUC: ${tm.pr_auc.toFixed(4)} (base 0.064)`),
        kpiCard("Tabular Sensibilidad OOF", (toof.recall_sensibilidad * 100).toFixed(1) + "%", `Umbral: ${toof.umbral} | Latencia: ${tab.prueba_inferencia.latencia_promedio_ms_50_runs} ms`)
      );

      $("tabEvidenceSummary").innerHTML = [
        evRow("Versión / Modelo", `${tab.identificacion.version} — ${tab.metodologia.modelo_seleccionado}`),
        evRow("Dimensiones", `${tab.datos.dimensiones_iniciales.filas}×${tab.datos.dimensiones_iniciales.columnas} → ${tab.datos.dimensiones_entrada_modelo.variables_entrada} variables sin fuga`),
        evRow("Desenlace Biopsy", `${tab.datos.desenlaces_registrados.Biopsy.positivos}/${tab.datos.dimensiones_iniciales.filas} (${tab.datos.desenlaces_registrados.Biopsy.prevalencia_pct}%)`),
        evRow("Validación", "StratifiedKFold (k=5, random_state=42)"),
        evRow("ROC-AUC / PR-AUC (CV)", `${tm.roc_auc} / ${tm.pr_auc}`),
        evRow("Recall / F1 (OOF thr*)", `${toof.recall_sensibilidad} / ${toof.f1_score}`),
      ].join("");
    }

    $("evidenceKpis").innerHTML = kpis.join("");

    const figs = [
      { title: "CNN — Matriz de Confusión 5×5 (Test)", src: "/outputs/cnn_confusion_matrix.png" },
      { title: "CNN — Curvas ROC y Precision-Recall (Test)", src: "/outputs/cnn_roc_pr_curves.png" },
      { title: "CNN — Historial de Entrenamiento", src: "/outputs/cnn_training_history.png" },
      { title: "CNN — Escalado de Anotaciones .dat a 224×224", src: "/outputs/cnn_sample_annotations_224.png" },
      { title: "CNN — Explicabilidad Grad-CAM (Muestras Test)", src: "/outputs/cnn_sample_gradcam.png" },
      { title: "Tabular — Matriz de Confusión OOF (5-Fold)", src: "/outputs/tabular_confusion_matrix.png" },
      { title: "Tabular — Curvas ROC y Precision-Recall (5-Fold)", src: "/outputs/tabular_roc_pr_curves.png" },
    ];

    $("figuresGrid").innerHTML = figs
      .map(
        (f) => `
        <div class="figure-card">
          <div class="figure-card-title">${f.title}</div>
          <img src="${f.src}" alt="${f.title}" onerror="this.closest('.figure-card').style.display='none'" />
        </div>
      `
      )
      .join("");
  } catch (err) {
    console.error(err);
  }
}

function kpiCard(label, val, sub) {
  return `<div class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-val">${val}</div><div class="kpi-sub">${sub}</div></div>`;
}
function evRow(k, v) {
  return `<div class="evidence-row"><span>${k}</span><strong>${v}</strong></div>`;
}

// Init
checkHealth();
loadSamples();
loadTabularSchema();
loadEvidence();
