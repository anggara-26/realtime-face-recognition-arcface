(function () {
  "use strict";

  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const videoBadge = document.getElementById("videoBadge");
  const engineSelect = document.getElementById("engineSelect");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const intervalMsInput = document.getElementById("intervalMs");
  const statusLine = document.getElementById("statusLine");
  const enrollForm = document.getElementById("enrollForm");
  const enrollName = document.getElementById("enrollName");
  const enrollFiles = document.getElementById("enrollFiles");
  const snapshotBtn = document.getElementById("snapshotBtn");
  const ttaCheckbox = document.getElementById("ttaCheckbox");
  const enrollStatus = document.getElementById("enrollStatus");
  const galleryList = document.getElementById("galleryList");

  const thresholds = (window.APP_CONFIG && window.APP_CONFIG.thresholds) || {};
  let loopHandle = null;
  let capturedSnapshotBlob = null;

  function getCsrfToken() {
    const match = document.cookie.match(/csrftoken=([^;]+)/);
    return match ? match[1] : "";
  }

  function setThresholdForEngine() {
    const t = thresholds[engineSelect.value] ?? 0.4;
    thresholdSlider.value = t;
    thresholdValue.textContent = Number(t).toFixed(2);
  }
  thresholdSlider.addEventListener("input", () => {
    thresholdValue.textContent = Number(thresholdSlider.value).toFixed(2);
  });
  engineSelect.addEventListener("change", () => {
    setThresholdForEngine();
    refreshGallery();
  });
  setThresholdForEngine();

  // --- Camera setup ----------------------------------------------------
  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      video.srcObject = stream;
      await new Promise((resolve) => (video.onloadedmetadata = resolve));
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      statusLine.textContent = "Kamera aktif. Siap mendaftarkan / mengenali wajah.";
    } catch (err) {
      statusLine.textContent = "Gagal mengakses kamera: " + err.message;
    }
  }
  initCamera();

  function captureFrameDataUrl(quality = 0.85) {
    const c = document.createElement("canvas");
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", quality);
  }

  function dataUrlToBlob(dataUrl) {
    const [meta, b64] = dataUrl.split(",");
    const mime = meta.match(/:(.*?);/)[1];
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  // --- Recognition loop --------------------------------------------------
  function drawDetections(faces) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    faces.forEach((f) => {
      const [x1, y1, x2, y2] = f.bbox;
      const known = f.label !== "Tak Dikenal";
      const color = known ? "#2ea043" : "#e5534b";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const text = `${f.label} ${f.score.toFixed(2)}`;
      ctx.font = "14px monospace";
      const textWidth = ctx.measureText(text).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 20), textWidth + 10, 20);
      ctx.fillStyle = "#0d1117";
      ctx.fillText(text, x1 + 5, Math.max(14, y1 - 5));
    });
  }

  async function recognizeOnce() {
    const image = captureFrameDataUrl();
    try {
      const res = await fetch("/api/recognize/", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": getCsrfToken() },
        body: JSON.stringify({
          image,
          engine: engineSelect.value,
          threshold: parseFloat(thresholdSlider.value),
        }),
      });
      const data = await res.json();
      if (data.ok) {
        drawDetections(data.faces);
        statusLine.textContent = `Terdeteksi ${data.faces.length} wajah (backend: ${data.engine}).`;
      } else {
        statusLine.textContent = "Error: " + data.error;
      }
    } catch (err) {
      statusLine.textContent = "Gagal menghubungi server: " + err.message;
    }
  }

  startBtn.addEventListener("click", () => {
    const interval = Math.max(150, parseInt(intervalMsInput.value, 10) || 700);
    loopHandle = setInterval(recognizeOnce, interval);
    startBtn.disabled = true;
    stopBtn.disabled = false;
    videoBadge.classList.add("live");
  });

  stopBtn.addEventListener("click", () => {
    clearInterval(loopHandle);
    loopHandle = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    videoBadge.classList.remove("live");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  });

  // --- Enrollment ----------------------------------------------------
  snapshotBtn.addEventListener("click", () => {
    const dataUrl = captureFrameDataUrl(0.92);
    capturedSnapshotBlob = dataUrlToBlob(dataUrl);
    enrollStatus.textContent = "Snapshot diambil dari kamera. Klik 'Daftarkan' untuk menyimpan.";
  });

  enrollForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("name", enrollName.value.trim());
    formData.append("engine", engineSelect.value);
    formData.append("tta", ttaCheckbox.checked ? "true" : "false");

    const files = enrollFiles.files;
    if (files.length > 0) {
      for (const f of files) formData.append("images", f);
    } else if (capturedSnapshotBlob) {
      formData.append("images", capturedSnapshotBlob, "snapshot.jpg");
    } else {
      enrollStatus.textContent = "Ambil snapshot atau pilih file foto terlebih dahulu.";
      return;
    }

    enrollStatus.textContent = "Memproses pendaftaran...";
    try {
      const res = await fetch("/api/enroll/", {
        method: "POST",
        headers: { "X-CSRFToken": getCsrfToken() },
        body: formData,
      });
      const data = await res.json();
      if (data.ok) {
        enrollStatus.textContent = `Berhasil mendaftarkan "${data.name}" (${data.num_source_photos} foto sumber).`;
        enrollForm.reset();
        capturedSnapshotBlob = null;
        refreshGallery();
      } else {
        enrollStatus.textContent = "Gagal: " + data.error;
      }
    } catch (err) {
      enrollStatus.textContent = "Gagal menghubungi server: " + err.message;
    }
  });

  // --- Gallery -------------------------------------------------------
  async function refreshGallery() {
    try {
      const res = await fetch(`/api/persons/?engine=${engineSelect.value}`);
      const data = await res.json();
      galleryList.innerHTML = "";
      if (!data.ok || data.persons.length === 0) {
        galleryList.innerHTML = '<li class="empty">Belum ada identitas terdaftar untuk backend ini.</li>';
        return;
      }
      data.persons.forEach((p) => {
        const li = document.createElement("li");
        const img = document.createElement("img");
        img.src = p.photo_url || "";
        img.alt = p.name;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `${p.name}<small>${p.engine} &middot; ${p.num_source_photos} foto</small>`;
        const delBtn = document.createElement("button");
        delBtn.textContent = "Hapus";
        delBtn.addEventListener("click", async () => {
          await fetch(`/api/persons/${p.id}/`, { method: "DELETE", headers: { "X-CSRFToken": getCsrfToken() } });
          refreshGallery();
        });
        li.append(img, meta, delBtn);
        galleryList.appendChild(li);
      });
    } catch (err) {
      galleryList.innerHTML = `<li class="empty">Gagal memuat galeri: ${err.message}</li>`;
    }
  }
  refreshGallery();
})();
