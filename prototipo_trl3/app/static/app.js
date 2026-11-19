// Prototipo IA-Tamizaje Cervical v0.1 (TRL 3) - Cliente Web Moderno y Responsive

const $ = (id) => document.getElementById(id);
const isMobileViewport = () => window.innerWidth <= 980;

// --- Lightbox Modal ---
function openLightbox(src, title) {
  if (!src) return;
  $("lightboxImg").src = src;
  $("lightboxTitle").textContent = title || "Vista ampliada";
  $("lightboxModal").classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  $("lightboxModal").classList.add("hidden");
  document.body.style.overflow = "";
}

$("lightboxClose").addEventListener("click", closeLightbox);
$("lightboxBackdrop").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

// --- Copy JSON helper ---
function setupCopyButton(btnId, preId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const txt = $(preId).textContent;
    try {
      await navigator.clipboard.writeText(txt);
      const prev = btn.textContent;
      btn.textContent = "Copiado";
      setTimeout(() => (btn.textContent = prev), 1400);
    } catch {
      btn.textContent = "No disponible";
    }
  });
}
setupCopyButton("btnCopyCnnJson", "cnnRawJson");
setupCopyButton("btnCopyTabJson", "tabRawJson");

// --- Tabs ---
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// --- Health check ---
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const pills = [
      data.cnn_ready
        ? `<span class="pill pill-ok"><span class="dot"></span>CNN MobileNetV2</span>`
        : `<span class="pill pill-warn"><span class="dot"></span>CNN pendiente</span>`,
      data.ood_ready
        ? `<span class="pill pill-ok"><span class="dot"></span>Filtro OOD</span>`
        : `<span class="pill pill-warn"><span class="dot"></span>OOD inactivo</span>`,
      data.tabular_ready
        ? `<span class="pill pill-ok"><span class="dot"></span>Tabular RF</span>`
        : `<span class="pill pill-warn"><span class="dot"></span>Tabular pendiente</span>`,
    ];
    $("statusPills").innerHTML = pills.join("");
  } catch {
    $("statusPills").innerHTML = `<span class="pill pill-warn"><span class="dot"></span>API desconectada</span>`;
  }
}

// --- CNN Upload & Samples ---
const dropzone = $("dropzone");
const imageInput = $("imageInput");

dropzone.addEventListener("click", () => imageInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    imageInput.click();
  }
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    clearActiveSampleButtons();
    sendImageFile(e.dataTransfer.files[0]);
  }
});
imageInput.addEventListener("change", () => {
  if (imageInput.files.length > 0) {
    clearActiveSampleButtons();
    sendImageFile(imageInput.files[0]);
  }
});

function clearActiveSampleButtons() {
  document.querySelectorAll(".sample-btn, .sample-btn-ood").forEach((b) => b.classList.remove("is-active"));
}

function setCnnLoading(loading, msg = "Procesando imagen, validando dominio OOD y generando Grad-CAM...") {
  $("cnnError").classList.add("hidden");
  if (loading) {
    $("cnnPlaceholder").classList.add("hidden");
    $("cnnResults").classList.add("hidden");
    $("cnnLoadingText").textContent = msg;
    $("cnnLoading").classList.remove("hidden");
    if (isMobileViewport()) {
      $("cnnResultCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } else {
    $("cnnLoading").classList.add("hidden");
  }
}

async function sendImageFile(file) {
  setCnnLoading(true, `Analizando "${file.name}"...`);
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch("/api/predict/image", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error en la predicción");
    setCnnLoading(false);
    renderCnnResult(data);
  } catch (err) {
    setCnnLoading(false);
    $("cnnPlaceholder").classList.remove("hidden");
    $("cnnError").textContent = err.message;
    $("cnnError").classList.remove("hidden");
  }
}

async function runSampleImage(filename, isOod = false, btnEl = null) {
  clearActiveSampleButtons();
  if (btnEl) btnEl.classList.add("is-active");

  setCnnLoading(true, isOod ? `Probando rechazo OOD sobre ${filename}...` : `Analizando muestra ${filename}...`);
  const endpoint = isOod
    ? `/api/predict/ood-sample/${encodeURIComponent(filename)}`
    : `/api/predict/sample/${encodeURIComponent(filename)}`;

  try {
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error al evaluar muestra");
    setCnnLoading(false);
    renderCnnResult(data);
  } catch (err) {
    setCnnLoading(false);
    $("cnnPlaceholder").classList.remove("hidden");
    $("cnnError").textContent = err.message;
    $("cnnError").classList.remove("hidden");
  }
}

function renderCnnResult(data) {
  $("cnnPlaceholder").classList.add("hidden");
  $("cnnResults").classList.remove("hidden");

  const p = data.prediccion;
  const ood = data.validacion_dominio_ood || {};
  const isValidDomain = data.dominio_valido !== false;

  const banner = $("cnnTriageBanner");
  banner.className = `triage-banner level-${p.nivel_tamizaje}`;

  if (!isValidDomain) {
    $("cnnTriageLabel").textContent = "FILTRO DE SEGURIDAD OOD";
    $("cnnPredTitle").textContent = "IMAGEN FUERA DE DOMINIO (RECHAZADA)";
    $("cnnPredDesc").textContent = p.recomendacion_apoyo;
    $("cnnAbnormalProb").textContent =
      ood.probabilidad_dominio_cervical !== undefined
        ? `${(ood.probabilidad_dominio_cervical * 100).toFixed(1)}%`
        : "OOD";
    $("cnnScoreCaption").textContent = "Prob. Cervical";
  } else {
    $("cnnTriageLabel").textContent = `PRIORIDAD DE TAMIZAJE: ${p.nivel_tamizaje}`;
    $("cnnPredTitle").textContent = `${p.nombre_clinico} (${(p.confianza * 100).toFixed(1)}%)`;
    $("cnnPredDesc").textContent = p.recomendacion_apoyo;
    $("cnnAbnormalProb").textContent = `${(p.probabilidad_anormal_agregada * 100).toFixed(1)}%`;
    $("cnnScoreCaption").textContent = "Prob. Anormal";
  }

  const oodChip = ood.habilitado
    ? `<span class="meta-chip ${isValidDomain ? "meta-chip-ok" : "meta-chip-ood"}">OOD: ${
        isValidDomain ? "Citología Válida" : "Rechazada"
      } (p=${ood.probabilidad_dominio_cervical}, cos=${ood.similitud_coseno_centroides}, ${ood.latencia_ood_ms} ms)</span>`
    : "";

  $("cnnMetaRow").innerHTML = `
    <span class="meta-chip">Archivo: ${data.entrada.archivo}</span>
    <span class="meta-chip">${data.entrada.tamano_original[0]}×${data.entrada.tamano_original[1]} → 224×224</span>
    <span class="meta-chip">CNN: ${data.tiempos_ms.inferencia_cnn} ms</span>
    ${isValidDomain ? `<span class="meta-chip">Grad-CAM: ${data.tiempos_ms.gradcam_xai} ms</span>` : ""}
    ${oodChip}
  `;

  $("cnnPreviewImg").src = data.imagenes.entrada_224;
  $("cnnPreviewZoom").onclick = () => openLightbox(data.imagenes.entrada_224, `Entrada Preprocesada (224×224) — ${data.entrada.archivo}`);

  const visualCompare = $("cnnVisualCompare");
  if (isValidDomain && data.imagenes.gradcam_overlay) {
    visualCompare.classList.remove("single-col");
    $("cnnGradcamBox").classList.remove("hidden");
    $("cnnGradcamImg").src = data.imagenes.gradcam_overlay;
    $("cnnGradcamZoom").onclick = () =>
      openLightbox(data.imagenes.gradcam_overlay, `Mapa Explicable Grad-CAM — ${p.nombre_clinico}`);

    $("cnnProbSection").classList.remove("hidden");
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
              <span class="prob-name"><strong>${item.nombre_es}</strong> <span class="muted">(${item.clase})</span></span>
              <strong class="prob-pct">${pct}%</strong>
            </div>
            <div class="prob-track">
              <div class="prob-fill ${clsType}" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      })
      .join("");
  } else {
    visualCompare.classList.add("single-col");
    $("cnnGradcamBox").classList.add("hidden");
    $("cnnProbSection").classList.add("hidden");
  }

  const cleanJson = {
    ...data,
    imagenes: {
      entrada_224: "[base64 PNG omitido]",
      gradcam_overlay: isValidDomain ? "[base64 PNG omitido]" : null,
    },
  };
  $("cnnRawJson").textContent = JSON.stringify(cleanJson, null, 2);
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();

    if (!data.samples || data.samples.length === 0) {
      $("sampleButtons").innerHTML = `<span class="muted">Sin muestras exportadas.</span>`;
    } else {
      $("sampleButtons").innerHTML = data.samples
        .map(
          (s) =>
            `<button type="button" class="sample-btn" data-file="${s.filename}">${s.expected_class
              .replace("im_", "")
              .replace("Superficial-Intermediate", "Superficial-Interm.")}</button>`
        )
        .join("");
      document.querySelectorAll(".sample-btn").forEach((btn) => {
        btn.addEventListener("click", () => runSampleImage(btn.dataset.file, false, btn));
      });
    }

    if (!data.ood_samples || data.ood_samples.length === 0) {
      $("oodSampleButtons").innerHTML = `<span class="muted">Sin muestras OOD disponibles.</span>`;
    } else {
      $("oodSampleButtons").innerHTML = data.ood_samples
        .map(
          (s) =>
            `<button type="button" class="sample-btn-ood" data-ood-file="${s.filename}">OOD: ${s.label}</button>`
        )
        .join("");
      document.querySelectorAll(".sample-btn-ood").forEach((btn) => {
        btn.addEventListener("click", () => runSampleImage(btn.dataset.oodFile, true, btn));
      });
    }
  } catch {
    $("sampleButtons").innerHTML = `<span class="muted">No se pudieron cargar las muestras.</span>`;
  }
}

// --- Tabular Form (Organized in Clean Collapsible Clinical Groups) ---
let tabularSchemaCache = null;

function renderFieldHtml(f) {
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
      <input type="number" inputmode="decimal" id="f_${f.name}" data-feature="${f.name}"
        min="${f.min ?? 0}" max="${f.max ?? 100}" step="${f.step ?? 1}"
        value="${f.default ?? f.median ?? 0}" placeholder="Mediana: ${f.median}" />
    </div>
  `;
}

async function loadTabularSchema() {
  try {
    const res = await fetch("/api/tabular/schema");
    if (!res.ok) return;
    tabularSchemaCache = await res.json();

    const groupDemo = [];
    const groupHabits = [];
    const groupStds = [];

    tabularSchemaCache.features.forEach((f) => {
      if (f.name.startsWith("STDs")) {
        groupStds.push(f);
      } else if (["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies"].includes(f.name)) {
        groupDemo.push(f);
      } else {
        groupHabits.push(f);
      }
    });

    const groups = [
      { title: "1. Demografía e Historia Reproductiva", items: groupDemo, open: true },
      { title: "2. Tabaquismo y Anticoncepción", items: groupHabits, open: true },
      { title: `3. Antecedentes de ETS (${groupStds.length} variables opcionales)`, items: groupStds, open: false },
    ];

    $("tabularSections").innerHTML = groups
      .map(
        (g) => `
        <details class="form-group-card" ${g.open ? "open" : ""}>
          <summary>${g.title}</summary>
          <div class="fields-grid">
            ${g.items.map(renderFieldHtml).join("")}
          </div>
        </details>
      `
      )
      .join("");
  } catch (err) {
    console.error(err);
  }
}

function applyTabularPreset(mode) {
  if (!tabularSchemaCache) return;
  const highValues = {
    Age: 38,
    "Number of sexual partners": 4,
    "First sexual intercourse": 18,
    "Num of pregnancies": 4,
    Smokes: 0,
    "Smokes (years)": 0,
    "Smokes (packs/year)": 0,
    "Hormonal Contraceptives": 1,
    "Hormonal Contraceptives (years)": 16,
    IUD: 1,
    "IUD (years)": 3,
    STDs: 0,
    "STDs (number)": 0,
    "STDs: Number of diagnosis": 0,
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
  const btn = $("btnSubmitTabular");
  const prevTxt = btn.textContent;
  btn.textContent = "Calculando...";
  btn.disabled = true;

  const values = {};
  document.querySelectorAll("[data-feature]").forEach((el) => {
    values[el.dataset.feature] = el.value === "" ? null : Number(el.value);
  });

  try {
    const res = await fetch("/api/predict/tabular", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error en inferencia tabular");
    renderTabularResult(data);
    if (isMobileViewport()) {
      $("tabResultCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    alert(err.message);
  } finally {
    btn.textContent = prevTxt;
    btn.disabled = false;
  }
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
    <span class="meta-chip">Imputadas: ${data.variables_imputadas}</span>
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
            <span class="prob-name">${f.variable}</span>
            <strong class="prob-pct">${pct}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${Math.min(100, pct * 3.2)}%"></div>
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
    const ood = ev.modulo_ood;

    if (cnn) {
      const m = cnn.metricas_test;
      const bin = m.tamizaje_clinico_binario;
      kpis.push(
        kpiCard("CNN Macro-F1", (m.macro_f1 * 100).toFixed(2) + "%", `Accuracy: ${(m.accuracy * 100).toFixed(2)}%`),
        kpiCard("CNN ROC-AUC", m.macro_roc_auc_ovr.toFixed(4), `PR-AUC: ${m.macro_pr_auc_ovr.toFixed(4)}`),
        kpiCard("Sensibilidad", (bin.sensibilidad_recall * 100).toFixed(2) + "%", `Especificidad: ${(bin.especificidad * 100).toFixed(2)}%`),
        kpiCard("Latencia CNN", cnn.prueba_inferencia.latencia_promedio_ms_30_runs + " ms", "0 láminas compartidas")
      );

      $("cnnEvidenceSummary").innerHTML = [
        evRow("Arquitectura", "MobileNetV2 (Transfer Learning)"),
        evRow("Dataset", `SIPaKMeD (${cnn.datos.total_imagenes} imágenes)`),
        evRow("Split sin fuga", `${cnn.datos.particion_sin_fuga.train.imagenes} / ${cnn.datos.particion_sin_fuga.val.imagenes} / ${cnn.datos.particion_sin_fuga.test.imagenes}`),
        evRow("Láminas compartidas", `${cnn.datos.particion_sin_fuga.laminas_compartidas_entre_splits} (verificado)`),
        evRow("Recall / Precision", `${cnn.metricas_test.macro_recall} / ${cnn.metricas_test.macro_precision}`),
      ].join("");
    }

    if (tab) {
      const tm = tab.metricas_modelo_seleccionado.cv_5fold;
      const toof = tab.metricas_modelo_seleccionado.oof_con_umbral_tamizaje;
      kpis.push(
        kpiCard("Tabular ROC-AUC", tm.roc_auc.toFixed(4), `PR-AUC: ${tm.pr_auc.toFixed(4)} (2.6× base)`),
        kpiCard("Tabular Esp.", (toof.especificidad * 100).toFixed(1) + "%", `Sensib.: ${(toof.recall_sensibilidad * 100).toFixed(1)}% | thr ${toof.umbral}`)
      );

      $("tabEvidenceSummary").innerHTML = [
        evRow("Modelo", tab.metodologia.modelo_seleccionado),
        evRow("Variables", `858×36 → ${tab.datos.dimensiones_entrada_modelo.variables_entrada} sin leakage`),
        evRow("Biopsy positivos", `${tab.datos.desenlaces_registrados.Biopsy.positivos}/858 (${tab.datos.desenlaces_registrados.Biopsy.prevalencia_pct}%)`),
        evRow("Validación", "StratifiedKFold (k=5)"),
        evRow("Latencia CPU", `${tab.prueba_inferencia.latencia_promedio_ms_50_runs} ms`),
      ].join("");
    }

    if (ood) {
      const om = ood.metricas_test;
      kpis.push(
        kpiCard("Rechazo OOD", (om.rechazo_no_cervical_especificidad * 100).toFixed(1) + "%", `Acepta citología: ${(om.aceptacion_citologia_valida_recall * 100).toFixed(1)}%`),
        kpiCard("Overhead OOD", `+${om.sobrecosto_latencia_ms} ms`, "Embeddings 1,280 dim")
      );

      $("oodEvidenceSummary").innerHTML = [
        evRow("Extractor", "MobileNetV2 compartido (1,280 dim)"),
        evRow("Negativos OOD", "archivebase2 (7 órganos, 21 subclases)"),
        evRow("Recall / Especif.", `${(om.aceptacion_citologia_valida_recall * 100).toFixed(1)}% / ${(om.rechazo_no_cervical_especificidad * 100).toFixed(1)}%`),
        evRow("Órganos no vistos", "96.4% – 100% (LOOO)"),
        evRow("Sintéticas no méd.", om.stress_test_sinteticas_rechazo),
      ].join("");
    }

    $("evidenceKpis").innerHTML = kpis.join("");

    const figs = [
      { title: "CNN — Matriz de Confusión 5×5 (Test)", src: "/outputs/cnn_confusion_matrix.png" },
      { title: "CNN — Curvas ROC y Precision-Recall (Test)", src: "/outputs/cnn_roc_pr_curves.png" },
      { title: "CNN — Historial de Entrenamiento", src: "/outputs/cnn_training_history.png" },
      { title: "CNN — Escalado de Anotaciones .dat a 224×224", src: "/outputs/cnn_sample_annotations_224.png" },
      { title: "CNN — Explicabilidad Grad-CAM (Muestras Test)", src: "/outputs/cnn_sample_gradcam.png" },
      { title: "Filtro OOD — Confusión y Leave-One-Organ-Out", src: "/outputs/ood_confusion_matrix.png" },
      { title: "Filtro OOD — Separación Scores y Coseno", src: "/outputs/ood_score_distribution.png" },
      { title: "Tabular — Matriz de Confusión OOF (5-Fold)", src: "/outputs/tabular_confusion_matrix.png" },
      { title: "Tabular — Curvas ROC y Precision-Recall (5-Fold)", src: "/outputs/tabular_roc_pr_curves.png" },
    ];

    $("figuresGrid").innerHTML = figs
      .map(
        (f) => `
        <article class="figure-card" data-zoom-src="${f.src}" data-zoom-title="${f.title}" tabindex="0" role="button">
          <div class="figure-card-title">${f.title}</div>
          <img src="${f.src}" alt="${f.title}" loading="lazy" onerror="this.closest('.figure-card').style.display='none'" />
        </article>
      `
      )
      .join("");

    document.querySelectorAll(".figure-card").forEach((card) => {
      const open = () => openLightbox(card.dataset.zoomSrc, card.dataset.zoomTitle);
      card.addEventListener("click", open);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      });
    });
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
