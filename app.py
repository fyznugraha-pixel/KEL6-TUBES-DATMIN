import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

# =====================================================
# LOAD MODEL DAN FITUR
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "student_status_simple_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "simple_features.pkl")

loaded_model = joblib.load(MODEL_PATH)
loaded_features = joblib.load(FEATURE_PATH)

app = Flask(__name__)

# =====================================================
# LABEL FITUR DAN HASIL PREDIKSI
# =====================================================

label_indonesia = {
    "Admission grade": "Nilai Seleksi Masuk Kampus",
    "Previous qualification (grade)": "Nilai Pendidikan Sebelumnya / Nilai Sekolah",
    "Age at enrollment": "Usia Saat Masuk Kuliah",
    "Tuition fees up to date": "Status Pembayaran UKT",
    "Curricular units 1st sem (enrolled)": "Jumlah Mata Kuliah Diambil Semester 1",
    "Curricular units 1st sem (approved)": "Jumlah Mata Kuliah Lulus Semester 1",
    "Curricular units 1st sem (grade)": "Nilai Rata-rata Semester 1",
    "Curricular units 2nd sem (enrolled)": "Jumlah Mata Kuliah Diambil Semester 2",
    "Curricular units 2nd sem (approved)": "Jumlah Mata Kuliah Lulus Semester 2",
    "Curricular units 2nd sem (grade)": "Nilai Rata-rata Semester 2"
}

label_prediksi = {
    "Graduate": "Lulus",
    "Dropout": "Putus Studi",
    "Enrolled": "Masih Aktif Kuliah"
}

feature_rows = []

for index, feature in enumerate(loaded_features):
    feature_rows.append({
        "feature": feature,
        "input_name": f"input_{index}",
        "label": label_indonesia.get(feature, feature)
    })

# =====================================================
# CSS GLOBAL
# =====================================================

BASE_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --primary-soft: #dbeafe;
        --navy: #0f172a;
        --text: #1e293b;
        --muted: #64748b;
        --border: #e2e8f0;
        --card: rgba(255, 255, 255, 0.86);
        --bg: #f8fafc;
        --success: #16a34a;
        --danger: #dc2626;
        --warning: #f59e0b;
    }

    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        font-family: 'Inter', Arial, sans-serif;
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 34%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.13), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
        color: var(--text);
        min-height: 100vh;
    }

    .navbar {
        width: 100%;
        position: sticky;
        top: 0;
        z-index: 50;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(226, 232, 240, 0.75);
    }

    .nav-inner {
        max-width: 1180px;
        margin: auto;
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        color: var(--navy);
        letter-spacing: -0.3px;
        font-size: 17px;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--primary), #38bdf8);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.28);
    }

    .nav-links {
        display: flex;
        gap: 8px;
        padding: 5px;
        border-radius: 999px;
        background: rgba(241, 245, 249, 0.9);
        border: 1px solid rgba(226, 232, 240, 0.9);
    }

    .nav-links a {
        text-decoration: none;
        color: var(--muted);
        font-weight: 700;
        font-size: 13px;
        padding: 9px 16px;
        border-radius: 999px;
        transition: 0.2s ease;
    }

    .nav-links a:hover {
        color: var(--primary);
        background: white;
    }

    .nav-links a.active {
        color: white;
        background: linear-gradient(135deg, var(--primary), #1d4ed8);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28);
    }

    .page {
        min-height: 100vh;
        padding: 42px 20px 70px;
    }

    .container {
        max-width: 1180px;
        margin: auto;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 38px;
        border-radius: 30px;
        color: white;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.94)),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.38), transparent 36%);
        box-shadow: 0 24px 60px rgba(30, 64, 175, 0.28);
        margin-bottom: 26px;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -90px;
        background: rgba(59, 130, 246, 0.28);
        border-radius: 50%;
        filter: blur(4px);
    }

    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 780px;
    }

    .tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 14px;
        margin-bottom: 18px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.13);
        color: #e0ecff;
        font-size: 13px;
        font-weight: 800;
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    h1 {
        margin: 0 0 14px;
        font-size: 42px;
        line-height: 1.08;
        letter-spacing: -1.3px;
    }

    .hero p {
        margin: 0;
        line-height: 1.8;
        color: #dbeafe;
        font-size: 16px;
    }

    .mini-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 26px;
        position: relative;
        z-index: 2;
    }

    .mini-stat {
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.11);
        border: 1px solid rgba(255, 255, 255, 0.13);
    }

    .mini-stat strong {
        display: block;
        font-size: 21px;
        margin-bottom: 3px;
    }

    .mini-stat span {
        color: #dbeafe;
        font-size: 12px;
        font-weight: 600;
    }

    .layout-grid {
        display: grid;
        grid-template-columns: 1.45fr 0.55fr;
        gap: 24px;
        align-items: start;
    }

    .card {
        background: var(--card);
        backdrop-filter: blur(18px);
        padding: 30px;
        border-radius: 26px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.10);
        border: 1px solid rgba(226, 232, 240, 0.88);
    }

    .card-title {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 22px;
    }

    .card-title h2 {
        margin: 0;
        font-size: 22px;
        color: var(--navy);
        letter-spacing: -0.5px;
    }

    .card-title p {
        margin: 7px 0 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.6;
    }

    .step-badge {
        padding: 8px 12px;
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary-dark);
        font-size: 12px;
        font-weight: 800;
        white-space: nowrap;
    }

    form {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 18px 22px;
    }

    .form-group {
        width: 100%;
    }

    label {
        display: block;
        margin-bottom: 8px;
        font-weight: 800;
        font-size: 13px;
        color: #334155;
        letter-spacing: -0.1px;
    }

    input,
    select {
        width: 100%;
        height: 48px;
        padding: 0 14px;
        border: 1px solid #dbe3ef;
        border-radius: 15px;
        font-size: 14px;
        outline: none;
        background: rgba(255, 255, 255, 0.92);
        color: var(--text);
        transition: 0.2s ease;
    }

    input:focus,
    select:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.12);
        background: white;
    }

    input::placeholder {
        color: #94a3b8;
    }

    .button-wrapper {
        grid-column: 1 / -1;
        margin-top: 8px;
    }

    button {
        width: 100%;
        padding: 16px;
        border: none;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--primary), #1d4ed8);
        color: white;
        font-size: 15px;
        font-weight: 800;
        cursor: pointer;
        transition: 0.2s ease;
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.24);
    }

    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 36px rgba(37, 99, 235, 0.30);
    }

    .side-panel {
        display: grid;
        gap: 16px;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.86);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
    }

    .info-icon {
        width: 42px;
        height: 42px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--primary-soft);
        color: var(--primary);
        font-size: 20px;
        margin-bottom: 13px;
    }

    .info-card h3 {
        margin: 0 0 8px;
        color: var(--navy);
        font-size: 16px;
    }

    .info-card p {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }

    .result {
        margin-top: 26px;
        padding: 24px;
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #bfdbfe;
        border-left: 7px solid var(--primary);
        border-radius: 22px;
    }

    .result h2 {
        margin: 0 0 10px;
        color: var(--navy);
        font-size: 24px;
    }

    .result p {
        color: var(--muted);
        line-height: 1.6;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 11px 17px;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--primary), #1d4ed8);
        color: white;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
    }

    .prob-list {
        margin-top: 22px;
    }

    .prob-list h3 {
        margin: 0 0 12px;
        color: var(--navy);
        font-size: 17px;
    }

    .prob-item {
        display: grid;
        grid-template-columns: 150px 1fr 70px;
        gap: 12px;
        align-items: center;
        padding: 11px 0;
        color: #334155;
    }

    .bar-track {
        height: 10px;
        border-radius: 999px;
        background: #dbeafe;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--primary), #38bdf8);
    }

    .prob-item strong {
        color: var(--primary-dark);
        text-align: right;
    }

    .result-layout {
        display: grid;
        grid-template-columns: 1.15fr 0.85fr;
        gap: 22px;
        align-items: center;
        margin-top: 18px;
    }

    .result-summary {
        display: flex;
        flex-direction: column;
        gap: 14px;
    }

    .result-chart-box {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .result-chart-box h3 {
        margin: 0 0 14px;
        color: var(--navy);
        font-size: 16px;
    }

    .result-chart-wrap {
        width: 100%;
        max-width: 260px;
        margin: auto;
    }

    .result-insight {
        margin-top: 14px;
        padding: 14px 16px;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .result-insight strong {
        color: var(--navy);
    }

    .input-summary {
        margin-top: 22px;
        padding: 18px;
        border-radius: 18px;
        background: rgba(248, 250, 252, 0.92);
        border: 1px solid #e2e8f0;
    }

    .input-summary h3 {
        margin: 0 0 14px;
        color: var(--navy);
        font-size: 17px;
    }

    .input-summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }

    .input-summary-item {
        padding: 12px 14px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e2e8f0;
    }

    .input-summary-item span {
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .input-summary-item strong {
        color: var(--navy);
        font-size: 14px;
    }

    .error {
        margin-top: 25px;
        padding: 16px;
        border-radius: 16px;
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        line-height: 1.5;
    }

    .footer-note {
        margin-top: 22px;
        padding: 15px 16px;
        border-radius: 16px;
        background: #f8fafc;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        border: 1px solid #e2e8f0;
    }

    .visual-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 24px;
        margin-top: 24px;
    }

    .visual-card {
        background: white;
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 14px 42px rgba(15, 23, 42, 0.09);
        border: 1px solid #e2e8f0;
    }

    .visual-card.full {
        grid-column: 1 / -1;
    }

    .visual-card h3 {
        margin: 0 0 8px;
        color: var(--navy);
        font-size: 18px;
    }

    .visual-card p {
        margin: 0 0 16px;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
    }

    .visual-card img {
        width: 100%;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 18px;
        margin-top: 24px;
    }

    .summary-card {
        background: rgba(255, 255, 255, 0.88);
        padding: 22px;
        border-radius: 22px;
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
    }

    .summary-card h3 {
        margin: 0 0 8px;
        color: var(--navy);
        font-size: 14px;
    }

    .summary-card .value {
        margin: 0;
        color: var(--primary);
        font-size: 30px;
        font-weight: 800;
    }

    .summary-card .desc {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    .custom-select {
        position: relative;
        width: 100%;
    }

    .custom-select-value {
        display: none;
    }

    .custom-select-trigger {
        width: 100%;
        height: 48px;
        padding: 0 14px;
        border: 1px solid #dbe3ef;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.92);
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: none;
        transition: 0.2s ease;
    }

    .custom-select-trigger:hover {
        transform: none;
        background: white;
        border-color: #2563eb;
        box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.10);
    }

    .custom-select.open .custom-select-trigger {
        border-color: #2563eb;
        box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.12);
    }

    .custom-select-trigger .selected-text {
        display: flex;
        align-items: center;
        gap: 9px;
    }

    .custom-select-trigger .arrow {
        font-size: 13px;
        transition: 0.2s ease;
    }

    .custom-select.open .arrow {
        transform: rotate(180deg);
    }

    .custom-select-menu {
        position: absolute;
        top: calc(100% + 8px);
        left: 0;
        right: 0;
        padding: 8px;
        border-radius: 17px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        display: none;
        z-index: 30;
    }

    .custom-select.open .custom-select-menu {
        display: block;
    }

    .custom-option {
        width: 100%;
        padding: 12px 13px;
        border: none;
        border-radius: 13px;
        background: transparent;
        color: #334155;
        font-size: 14px;
        font-weight: 700;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 10px;
        text-align: left;
        transition: 0.18s ease;
        box-shadow: none;
    }

    .custom-option:hover {
        transform: none;
        background: #eff6ff;
        color: #1d4ed8;
    }

    .option-icon {
        width: 28px;
        height: 28px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }

    .option-paid .option-icon {
        background: #dcfce7;
        color: #16a34a;
    }

    .option-unpaid .option-icon {
        background: #fee2e2;
        color: #dc2626;
    }

    .custom-select.error .custom-select-trigger {
        border-color: #ef4444;
        box-shadow: 0 0 0 5px rgba(239, 68, 68, 0.10);
    }

    @media (max-width: 980px) {
        .layout-grid {
            grid-template-columns: 1fr;
        }

        .summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .visual-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 768px) {
        .nav-inner {
            flex-direction: column;
            gap: 14px;
        }

        .page {
            padding: 28px 14px 55px;
        }

        .hero {
            padding: 28px;
            border-radius: 24px;
        }

        h1 {
            font-size: 31px;
        }

        .mini-stats {
            grid-template-columns: 1fr;
        }

        form {
            grid-template-columns: 1fr;
        }

        .card {
            padding: 22px;
            border-radius: 22px;
        }

        .card-title {
            flex-direction: column;
        }

        .prob-item {
            grid-template-columns: 1fr;
            gap: 7px;
        }

        .prob-item strong {
            text-align: left;
        }

        .summary-grid {
            grid-template-columns: 1fr;
        }

        .result-layout {
            grid-template-columns: 1fr;
        }

        .result-chart-wrap {
            max-width: 210px;
        }

        .input-summary-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""

NAVBAR = """
<div class="navbar">
    <div class="nav-inner">
        <div class="brand">
            <div class="brand-mark">D</div>
            <span>DATMIN KEL 6</span>
        </div>

        <div class="nav-links">
            <a href="/" class="{% if active_page == 'form' %}active{% endif %}">Form Prediksi</a>
            <a href="/visualisasi" class="{% if active_page == 'visualisasi' %}active{% endif %}">Visualisasi</a>
        </div>
    </div>
</div>
"""

# =====================================================
# HALAMAN FORM PREDIKSI
# =====================================================

FORM_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Prediksi Status Mahasiswa</title>
    """ + BASE_STYLE + """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>
    """ + NAVBAR + """

    <div class="page">
        <div class="container">

            <div class="hero">
                <div class="hero-content">
                    <span class="tag">● Deployment Model Data Mining</span>
                    <h1>Prediksi Status Mahasiswa</h1>
                    <p>
                        Sistem ini membantu memprediksi kemungkinan status mahasiswa berdasarkan data akademik
                        semester awal. Hasil prediksi terdiri dari <b>Lulus</b>, <b>Putus Studi</b>, atau
                        <b>Masih Aktif Kuliah</b>.
                    </p>
                </div>

                <div class="mini-stats">
                    <div class="mini-stat">
                        <strong>3</strong>
                        <span>Kategori status</span>
                    </div>

                    <div class="mini-stat">
                        <strong>{{ total_features }}</strong>
                        <span>Fitur input model</span>
                    </div>

                    <div class="mini-stat">
                        <strong>ML</strong>
                        <span>Classification model</span>
                    </div>
                </div>
            </div>

            <div class="layout-grid">

                <div class="card">
                    <div class="card-title">
                        <div>
                            <h2>Masukkan Data Mahasiswa</h2>
                            <p>
                                Isi data sesuai format angka pada dataset. Untuk status pembayaran UKT,
                                pilih sudah bayar atau belum bayar.
                            </p>
                        </div>
                        <span class="step-badge">Step 1 / Prediksi</span>
                    </div>

                    <form method="POST" action="/predict">
                        {% for item in feature_rows %}
                        <div class="form-group">
                            <label for="{{ item.input_name }}">{{ item.label }}</label>

                            {% if item.feature == "Tuition fees up to date" %}
                            {% set selected_val = submitted_values.get(item.input_name, '') %}

                            <div class="custom-select" data-select>
                                <input
                                    type="hidden"
                                    id="{{ item.input_name }}"
                                    name="{{ item.input_name }}"
                                    class="custom-select-value"
                                    value="{{ selected_val }}">

                                <button type="button" class="custom-select-trigger">
                                    <span class="selected-text">
                                        {% if selected_val == "1" %}
                                            <span>✅</span>
                                            <span>Sudah Bayar</span>
                                        {% elif selected_val == "0" %}
                                            <span>⚠️</span>
                                            <span>Belum Bayar</span>
                                        {% else %}
                                            <span>💳</span>
                                            <span>Pilih status pembayaran</span>
                                        {% endif %}
                                    </span>
                                    <span class="arrow">⌄</span>
                                </button>

                                <div class="custom-select-menu">
                                    <button type="button" class="custom-option option-paid" data-value="1" data-label="Sudah Bayar">
                                        <span class="option-icon">✓</span>
                                        <span>Sudah Bayar</span>
                                    </button>

                                    <button type="button" class="custom-option option-unpaid" data-value="0" data-label="Belum Bayar">
                                        <span class="option-icon">!</span>
                                        <span>Belum Bayar</span>
                                    </button>
                                </div>
                            </div>
                            {% else %}
                            <input
                                type="number"
                                step="any"
                                id="{{ item.input_name }}"
                                name="{{ item.input_name }}"
                                placeholder="Masukkan nilai"
                                value="{{ submitted_values.get(item.input_name, '') }}"
                                required>
                            {% endif %}
                        </div>
                        {% endfor %}

                        <div class="button-wrapper">
                            <button type="submit">Prediksi Status Mahasiswa</button>
                        </div>
                    </form>

                    {% if prediction %}
                    <div class="result">
                        <h2>Hasil Prediksi</h2>
                        <p>Berdasarkan data yang dimasukkan, status mahasiswa diprediksi sebagai:</p>

                        <div class="result-layout">

                            <div class="result-summary">
                                <span class="badge">● {{ prediction }}</span>

                                <div class="prob-list">
                                    <h3>Probabilitas Tiap Kelas</h3>

                                    {% for cls, value in probabilities.items() %}
                                    <div class="prob-item">
                                        <span>{{ cls }}</span>
                                        <div class="bar-track">
                                            <div class="bar-fill" style="width: {{ value }}%;"></div>
                                        </div>
                                        <strong>{{ value }}%</strong>
                                    </div>
                                    {% endfor %}
                                </div>

                                <div class="result-insight">
                                    <strong>Insight singkat:</strong><br>
                                    Grafik di samping menunjukkan distribusi probabilitas dari ketiga kelas.
                                    Semakin besar persentase pada satu kelas, semakin tinggi keyakinan model terhadap hasil prediksi tersebut.
                                </div>
                            </div>

                            <div class="result-chart-box">
                                <h3>Visualisasi Hasil Prediksi</h3>
                                <div class="result-chart-wrap">
                                    <canvas id="predictionChart"></canvas>
                                </div>
                            </div>

                        </div>

                        <div class="input-summary">
                            <h3>Data Input yang Digunakan</h3>

                            <div class="input-summary-grid">
                                {% for item in input_summary %}
                                <div class="input-summary-item">
                                    <span>{{ item.label }}</span>
                                    <strong>{{ item.value }}</strong>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>

                    <script>
                        const predictionCanvas = document.getElementById('predictionChart');

                        if (predictionCanvas) {
                            new Chart(predictionCanvas, {
                                type: 'doughnut',
                                data: {
                                    labels: [
                                        {% for cls, value in probabilities.items() %}
                                        "{{ cls }}"{% if not loop.last %},{% endif %}
                                        {% endfor %}
                                    ],
                                    datasets: [{
                                        data: [
                                            {% for cls, value in probabilities.items() %}
                                            {{ value }}{% if not loop.last %},{% endif %}
                                            {% endfor %}
                                        ],
                                        backgroundColor: [
                                            '#ef4444',
                                            '#f59e0b',
                                            '#2563eb'
                                        ],
                                        borderWidth: 0,
                                        hoverOffset: 8
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    cutout: '68%',
                                    plugins: {
                                        legend: {
                                            position: 'bottom',
                                            labels: {
                                                boxWidth: 12,
                                                padding: 14,
                                                font: {
                                                    size: 12,
                                                    family: 'Inter'
                                                }
                                            }
                                        },
                                        tooltip: {
                                            callbacks: {
                                                label: function(context) {
                                                    return context.label + ': ' + context.raw + '%';
                                                }
                                            }
                                        }
                                    }
                                }
                            });
                        }
                    </script>
                    {% endif %}

                    {% if error %}
                    <div class="error">
                        {{ error }}
                    </div>
                    {% endif %}

                    <div class="footer-note">
                        Catatan: nilai input mengikuti format dataset training.
                        Nilai rata-rata semester menggunakan skala angka yang sama seperti data asli.
                    </div>
                </div>

                <div class="side-panel">
                    <div class="info-card">
                        <div class="info-icon">📌</div>
                        <h3>Tujuan Sistem</h3>
                        <p>
                            Sistem ini dibuat untuk menerapkan proses data mining dari tahap preprocessing,
                            modeling, evaluasi, sampai deployment sederhana berbasis web.
                        </p>
                    </div>

                    <div class="info-card">
                        <div class="info-icon">🎯</div>
                        <h3>Target Prediksi</h3>
                        <p>
                            Target model terdiri dari tiga kelas: Lulus, Putus Studi, dan Masih Aktif Kuliah.
                            Karena targetnya berupa kategori, metode utama yang digunakan adalah classification.
                        </p>
                    </div>

                    <div class="info-card">
                        <div class="info-icon">💡</div>
                        <h3>Tips Input</h3>
                        <p>
                            Untuk UKT pilih “Sudah Bayar” jika pembayaran up to date. Jika belum sesuai,
                            pilih “Belum Bayar”. Input lain diisi menggunakan angka sesuai data mahasiswa.
                        </p>
                    </div>
                </div>

            </div>

        </div>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function () {
            const customSelects = document.querySelectorAll("[data-select]");

            customSelects.forEach(function (selectBox) {
                const trigger = selectBox.querySelector(".custom-select-trigger");
                const selectedText = selectBox.querySelector(".selected-text");
                const hiddenInput = selectBox.querySelector(".custom-select-value");
                const options = selectBox.querySelectorAll(".custom-option");

                trigger.addEventListener("click", function () {
                    customSelects.forEach(function (otherSelect) {
                        if (otherSelect !== selectBox) {
                            otherSelect.classList.remove("open");
                        }
                    });

                    selectBox.classList.toggle("open");
                });

                options.forEach(function (option) {
                    option.addEventListener("click", function () {
                        const value = option.getAttribute("data-value");
                        const label = option.getAttribute("data-label");

                        hiddenInput.value = value;

                        selectedText.innerHTML = `
                            <span>${value === "1" ? "✅" : "⚠️"}</span>
                            <span>${label}</span>
                        `;

                        selectBox.classList.remove("open");
                        selectBox.classList.remove("error");
                    });
                });
            });

            document.addEventListener("click", function (event) {
                customSelects.forEach(function (selectBox) {
                    if (!selectBox.contains(event.target)) {
                        selectBox.classList.remove("open");
                    }
                });
            });

            const form = document.querySelector("form");

            form.addEventListener("submit", function (event) {
                const customSelect = document.querySelector("[data-select]");

                if (!customSelect) {
                    return;
                }

                const hiddenInput = customSelect.querySelector(".custom-select-value");

                if (!hiddenInput.value) {
                    event.preventDefault();
                    customSelect.classList.add("error");
                    alert("Status pembayaran UKT harus dipilih dulu.");
                }
            });
        });
    </script>
</body>
</html>
"""

# =====================================================
# HALAMAN VISUALISASI
# =====================================================

VISUALISASI_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Visualisasi Data Mining</title>
    """ + BASE_STYLE + """
</head>

<body>
    """ + NAVBAR + """

    <div class="page">
        <div class="container">

            <div class="hero">
                <span class="tag">Dashboard Visualisasi</span>
                <h1>Visualisasi Hasil Data Mining</h1>
                <p>
                    Halaman ini menampilkan visualisasi utama dari notebook Colab.
                    Visualisasi yang ditampilkan sudah dipilih agar dashboard lebih ringkas,
                    tidak terlalu penuh, dan tetap relevan untuk menjelaskan proses analisis data mining.
                </p>
            </div>

            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Data</h3>
                    <p class="value">4.424</p>
                    <p class="desc">Jumlah data mahasiswa pada dataset.</p>
                </div>

                <div class="summary-card">
                    <h3>Total Fitur Awal</h3>
                    <p class="value">37</p>
                    <p class="desc">Jumlah kolom awal sebelum feature selection.</p>
                </div>

                <div class="summary-card">
                    <h3>Akurasi Model Utama</h3>
                    <p class="value">77,51%</p>
                    <p class="desc">Akurasi Random Forest pada model utama.</p>
                </div>

                <div class="summary-card">
                    <h3>Fitur Deployment</h3>
                    <p class="value">{{ total_features }}</p>
                    <p class="desc">Jumlah fitur yang digunakan pada form prediksi.</p>
                </div>
            </div>

            <div class="visual-grid">

                <div class="visual-card">
                    <h3>Distribusi Status Mahasiswa</h3>
                    <p>
                        Grafik ini menunjukkan jumlah data pada setiap status mahasiswa,
                        yaitu Lulus, Putus Studi, dan Masih Aktif Kuliah.
                    </p>
                    <img src="/static/img/01_distribusi_target.png" alt="Distribusi Target">
                </div>

                <div class="visual-card">
                    <h3>Feature Importance</h3>
                    <p>
                        Grafik ini menunjukkan fitur yang paling berpengaruh terhadap prediksi status mahasiswa
                        berdasarkan model Random Forest.
                    </p>
                    <img src="/static/img/03_feature_importance.png" alt="Feature Importance">
                </div>

                <div class="visual-card full">
                    <h3>Correlation Heatmap</h3>
                    <p>
                        Heatmap digunakan untuk melihat hubungan antar variabel numerik.
                        Dari sini dapat dilihat fitur mana yang memiliki hubungan cukup kuat dengan fitur lainnya.
                    </p>
                    <img src="/static/img/02_heatmap_korelasi.png" alt="Correlation Heatmap">
                </div>

                <div class="visual-card">
                    <h3>Elbow Method</h3>
                    <p>
                        Elbow Method digunakan untuk membantu melihat jumlah cluster yang cukup optimal
                        berdasarkan penurunan nilai inertia.
                    </p>
                    <img src="/static/img/04_elbow_method.png" alt="Elbow Method">
                </div>

                <div class="visual-card">
                    <h3>Silhouette Score</h3>
                    <p>
                        Silhouette Score digunakan untuk melihat kualitas pemisahan cluster.
                        Nilai yang lebih tinggi menunjukkan cluster yang lebih baik.
                    </p>
                    <img src="/static/img/05_silhouette_score.png" alt="Silhouette Score">
                </div>

                <div class="visual-card">
                    <h3>Visualisasi Cluster dengan PCA</h3>
                    <p>
                        PCA digunakan untuk menurunkan dimensi data menjadi dua dimensi
                        agar hasil clustering dapat divisualisasikan.
                    </p>
                    <img src="/static/img/06_cluster_pca.png" alt="Cluster PCA">
                </div>

                <div class="visual-card">
                    <h3>Confusion Matrix Random Forest</h3>
                    <p>
                        Confusion matrix menunjukkan perbandingan antara label asli dan hasil prediksi model.
                        Grafik ini membantu melihat kelas mana yang sudah baik dan mana yang masih sering tertukar.
                    </p>
                    <img src="/static/img/07_confusion_matrix_random_forest.png" alt="Confusion Matrix Random Forest">
                </div>

            </div>

            <div class="footer-note">
                Catatan: visualisasi pada halaman ini diambil dari output notebook Colab.
                Grafik yang terlalu banyak atau kurang relevan tidak ditampilkan agar dashboard lebih fokus dan mudah dipahami saat presentasi.
            </div>

        </div>
    </div>
</body>
</html>
"""

# =====================================================
# ROUTE HALAMAN FORM
# =====================================================

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        FORM_TEMPLATE,
        active_page="form",
        feature_rows=feature_rows,
        total_features=len(loaded_features),
        prediction=None,
        probabilities=None,
        input_summary=[],
        submitted_values={},
        error=None
    )

# =====================================================
# ROUTE PREDIKSI
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []
        submitted_values = {}
        input_summary = []

        for item in feature_rows:
            raw_value = request.form[item["input_name"]]
            submitted_values[item["input_name"]] = raw_value

            value = float(raw_value)
            input_values.append(value)

            display_value = raw_value

            if item["feature"] == "Tuition fees up to date":
                if raw_value == "1":
                    display_value = "Sudah Bayar"
                elif raw_value == "0":
                    display_value = "Belum Bayar"

            input_summary.append({
                "label": item["label"],
                "value": display_value
            })

        input_df = pd.DataFrame([input_values], columns=loaded_features)

        prediction_asli = loaded_model.predict(input_df)[0]
        prediction = label_prediksi.get(prediction_asli, prediction_asli)

        proba = loaded_model.predict_proba(input_df)[0]

        probabilities = {
            label_prediksi.get(loaded_model.classes_[i], loaded_model.classes_[i]): round(float(proba[i]) * 100, 2)
            for i in range(len(loaded_model.classes_))
        }

        return render_template_string(
            FORM_TEMPLATE,
            active_page="form",
            feature_rows=feature_rows,
            total_features=len(loaded_features),
            prediction=prediction,
            probabilities=probabilities,
            input_summary=input_summary,
            submitted_values=submitted_values,
            error=None
        )

    except Exception as e:
        return render_template_string(
            FORM_TEMPLATE,
            active_page="form",
            feature_rows=feature_rows,
            total_features=len(loaded_features),
            prediction=None,
            probabilities=None,
            input_summary=[],
            submitted_values=dict(request.form),
            error=f"Terjadi error saat melakukan prediksi: {e}"
        )

# =====================================================
# ROUTE HALAMAN VISUALISASI
# =====================================================

@app.route("/visualisasi", methods=["GET"])
def visualisasi():
    return render_template_string(
        VISUALISASI_TEMPLATE,
        active_page="visualisasi",
        total_features=len(loaded_features)
    )

# =====================================================
# LOCAL TESTING
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
