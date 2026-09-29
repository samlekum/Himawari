# pipeline/visualization.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = BASE_DIR /"KotaBandung.geojson"

BANDUNG_BOUNDARY = gpd.read_file(GEOJSON_PATH)
# Garis luar kota (dissolve semua kecamatan jadi satu polygon), dihitung sekali
# saat modul di-load supaya tidak diulang tiap render_six_panel dipanggil.
BANDUNG_OUTLINE = BANDUNG_BOUNDARY.dissolve()

# ==== Threshold & Color Scheme (Modern Dark) ====
CLOUD_BINS = [220, 260]
CLOUD_LABELS = ["Hujan", "Mendung", "Tidak Hujan"]
CLOUD_COLORS = [
    "#08285B",
    "#6A727D",
    "#B89300",
]

RISK_BINS = [240, 265]
RISK_LABELS = ["Bahaya", "Waspada", "Aman"]
RISK_COLORS = [
    "#9B0000",
    "#B89300",
    "#217C00",
]

GRAD_TOP    = "#06111B"
GRAD_BOTTOM = "#204A73"

CARD_FACE = "#162434"
CARD_ALPHA = 0.78
CARD_EDGE = "#35516B"

FG_COLOR = "#F8FAFC"
MUTED_COLOR = "#C7D5E5"

GLASS_THERMAL = LinearSegmentedColormap.from_list(
    "premium",
    [
        "#08245E",
        "#092762",
        "#0A2B68",
        "#0B2F6E",
        "#0C3375",
        "#0E387D",
        "#104086",
        "#12488F",
        "#15519B",
        "#185AA8",
        "#1D66B8",
        "#2473CA",
        "#2F82DE",
        "#3F93F1",
        "#52A6FF",
        "#68BBFF",
        "#82D1FF",
        "#9EE6FF",
        "#B9F5FF",

        "#A3E68E",
        "#8ED97D",
        "#7FC97A",
        "#92CF71",
        "#A7D46A",
        "#BDD761",
        "#D1D95A",
        "#E4D853",

        "#F1CF47",
        "#F4C33C",
        "#F6B532",
        "#F8A429",
        "#F88F23",
        "#F67A20",
        "#F1661E",

        "#EB521C",
        "#E23F1A",
        "#D62E18",
        "#C91D1D",

        "#B31617",
        "#9B1010",
        "#800A0A",
    ],
    N=1024,
)

GLASS_ERROR = LinearSegmentedColormap.from_list( 
    "glass_error", [ 
        "#F7F3F3", 
        "#DFDFDF", 
        "#FFEB75",
        "#FFD500",
        "#D07B2F",
        "#8A2A17",
        "#490707",
        ]
    )

# ==== Skala CTT (K) & Error (K) yang FIKS, supaya tidak berubah-ubah antar panel/run ====
CTT_VMIN = 220.0
CTT_VMAX = 300.0
CTT_LEVELS = np.linspace(CTT_VMIN, CTT_VMAX, 161)
CTT_TICKS = np.arange(220, 301, 10)

ERROR_VMIN = 0.0
ERROR_VMAX = 10.0
ERROR_LEVELS = np.linspace(ERROR_VMIN, ERROR_VMAX, 21)
ERROR_TICKS = np.linspace(ERROR_VMIN, ERROR_VMAX, 6)


def classify(values, bins):
    """Klasifikasi array nilai CTT ke indeks kategori (0..n-1) berdasarkan bins."""
    return np.digitize(values, bins)


def pivot_grid(df_frame, value_col, n_lat=5, n_lon=7):
    """Ubah dataframe panjang (pixel_row, pixel_col, value) jadi array 2D (n_lat, n_lon)."""
    grid = np.full((n_lat, n_lon), np.nan)
    for row in df_frame.itertuples(index=False):
        grid[int(row.pixel_row), int(row.pixel_col)] = getattr(row, value_col)
    return grid


def get_axis_arrays(df_frame, n_lat=5, n_lon=7):
    """Ambil array lat (n_lat,) & lon (n_lon,) asli dari data (urutan sesuai pixel_row/pixel_col)."""
    lat_arr = np.zeros(n_lat)
    lon_arr = np.zeros(n_lon)
    for i in range(n_lat):
        lat_arr[i] = df_frame.loc[df_frame["pixel_row"] == i, "lat"].iloc[0]
    for j in range(n_lon):
        lon_arr[j] = df_frame.loc[df_frame["pixel_col"] == j, "lon"].iloc[0]
    return lat_arr, lon_arr


def interpolate_grid(lat_arr, lon_arr, values_2d, factor=24):
    """Upsample grid kasar (5x7) jadi lebih halus pakai interpolasi kubik, biar kontur mulus."""
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    vals = values_2d.ravel()

    valid = ~np.isnan(vals)
    lon_fine = np.linspace(lon_arr.min(), lon_arr.max(), len(lon_arr) * factor)
    lat_fine = np.linspace(lat_arr.min(), lat_arr.max(), len(lat_arr) * factor)
    lon_fine_grid, lat_fine_grid = np.meshgrid(lon_fine, lat_fine)

    vals_fine = griddata(points[valid], vals[valid], (lon_fine_grid, lat_fine_grid), method="cubic")
    vals_fine = np.clip(
        vals_fine,
        CTT_VMIN,
        CTT_VMAX,
    )
    if np.isnan(vals_fine).any():
        vals_nearest = griddata(points[valid], vals[valid], (lon_fine_grid, lat_fine_grid), method="nearest")
        vals_fine = np.where(np.isnan(vals_fine), vals_nearest, vals_fine)

    return lat_fine, lon_fine, vals_fine


def _add_gradient_background(fig):
    """Background gradient dark modern."""
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-20)
    bg_ax.set_axis_off()
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("bg_grad", [GRAD_TOP, GRAD_BOTTOM])
    bg_ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1], origin="upper")
    return bg_ax


def _draw_kecamatan(ax):
    """
    Menggambar batas kecamatan dan nama kecamatan.
    """

    # Garis batas antar-kecamatan (tipis & redup, cuma buat referensi internal)
    BANDUNG_BOUNDARY.boundary.plot(
        ax=ax,
        edgecolor="#F5F5F5",
        linewidth=0.55,
        alpha=0.45,
        zorder=14,
    )

    # Garis batas luar Kota Bandung (tebal & terang, biar langsung kebaca
    # sebagai batas kota -- bukan cuma batas kecamatan biasa)
    BANDUNG_OUTLINE.boundary.plot(
        ax=ax,
        edgecolor="#FFFFFF",
        linewidth=2.4,
        alpha=1.0,
        zorder=16,
        path_effects=[pe.withStroke(linewidth=3.6, foreground="#0B2F6E", alpha=0.65)],
    )

    # Label "KOTA BANDUNG" besar & redup di area kota, di belakang label
    # kecamatan, biar konteks kota langsung kebaca sekilas.
    minx, miny, maxx, maxy = BANDUNG_OUTLINE.total_bounds
    ax.text(
        (minx + maxx) / 2,
        maxy - (maxy - miny) * 0.06,
        "KOTA BANDUNG",
        fontsize=8.5,
        fontweight="bold",
        alpha=0.55,
        color="#FFFFFF",
        ha="center",
        va="top",
        style="italic",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#000000", alpha=0.5)],
        zorder=13,
    )

    # Nama kecamatan
    for _, row in BANDUNG_BOUNDARY.iterrows():

        point = row.geometry.representative_point()
        name = row["NAME_3"].replace(" ", "\n", 1)

        ax.text(
            point.x,
            point.y,

            name,

            fontsize=6,
            fontweight="bold",
            alpha=0.9,
            clip_on=True,

            path_effects=[
                pe.withStroke(
                    linewidth=1.4,
                    foreground="#000000"
                )
            ],

            color="white",

            ha="center",
            va="center",

            zorder=10,
        )


def _style_axis(ax, lat_fine, lon_fine):
    ax.set_facecolor("none")
    minx, miny, maxx, maxy = BANDUNG_BOUNDARY.total_bounds

    margin_x = (maxx - minx) * 0.03
    margin_y = (maxy - miny) * 0.03

    ax.set_xlim(minx - margin_x, maxx + margin_x)
    ax.set_ylim(miny - margin_y, maxy + margin_y)

    _draw_kecamatan(ax)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=MUTED_COLOR, labelsize=6.5, length=0)
    ax.tick_params(axis="y", labelrotation=45)

    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_continuous( ax, lat_fine, lon_fine, values_fine, title, cmap=GLASS_THERMAL, vmin=CTT_VMIN, vmax=CTT_VMAX, levels=CTT_LEVELS, ticks=CTT_TICKS, cb_label="CTT (K)"):
    ax.set_title(
        title,
        color=FG_COLOR,
        fontsize=12,
        fontweight="700",
        pad=8,
    )

    norm = plt.Normalize(vmin, vmax)

    cf = ax.contourf(
        lon_fine,
        lat_fine,
        values_fine,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        antialiased=True,
    )

    cb = plt.colorbar(
        cf,
        ax=ax,
        fraction=0.046,
        pad=0.03,
        shrink=0.83,
        aspect=22,
        ticks=ticks,
    )

    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
    cb.set_label(cb_label, color=MUTED_COLOR, fontsize=7)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    _style_axis(ax, lat_fine, lon_fine)

    return cb

def _add_category_legend(ax, labels, colors):
    patches = [mpatches.Patch(facecolor=c, edgecolor="none", label=l, alpha=0.95) for l, c in zip(labels, colors)]
    leg = ax.legend(
        handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=len(labels), frameon=True, fontsize=8, handlelength=1.6, borderpad=0.8, labelspacing=0.8,
        columnspacing=1.2, handletextpad=0.6
    )
    leg.get_frame().set_facecolor(CARD_FACE)
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor(CARD_EDGE)
    for text in leg.get_texts():
        text.set_color(FG_COLOR)


def _plot_categorical(ax, lat_fine, lon_fine, values_fine, bins, labels, colors, title):
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
    cat = classify(values_fine, bins)
    cat_smooth = gaussian_filter(cat.astype(float), sigma=2)
    cmap = LinearSegmentedColormap.from_list(
        "cloud",
        colors,
        N=512,
    )
    norm = BoundaryNorm(list(range(len(labels) + 1)), cmap.N)

    cf = ax.contourf(
        lon_fine,
        lat_fine,
        cat_smooth,
        levels=np.arange(len(labels)+1)-0.5,
        cmap=cmap,
        alpha=0.96,
        antialiased=True,
    )

    ax.contour(
        lon_fine,
        lat_fine,
        cat,
        levels=np.arange(len(labels)+1)-0.5,
        colors="white",
        linewidths=0.18,
        alpha=0.12,
    )

    _style_axis(ax, lat_fine, lon_fine)
    _add_category_legend(ax, labels, colors)
    return cat


def _plot_placeholder(
    ax, lat_fine, lon_fine, title,
    kind="continuous", cmap=GLASS_THERMAL, vmin=CTT_VMIN, vmax=CTT_VMAX,
    ticks=CTT_TICKS, cb_label="CTT (K)", labels=None, colors=None,
):
    """
    Placeholder panel saat data aktual belum tersedia.
    Tetap menggambar colorbar/legend kosong dengan geometri sama
    seperti panel berisi data, supaya ukuran subplot tetap konsisten.
    """
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="700", pad=8)

    if kind == "continuous":
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, shrink=0.83, aspect=22, ticks=ticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
        cb.set_label(cb_label, color=MUTED_COLOR, fontsize=7)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    _style_axis(ax, lat_fine, lon_fine)

    if kind == "categorical":
        _add_category_legend(ax, labels, colors)

    ax.text(
        0.5, 0.5, "Data aktual\nbelum tersedia",
        color=MUTED_COLOR, fontsize=12, fontweight="700", ha="center", va="center",
        transform=ax.transAxes, zorder=20,
    )


def render_six_panel(
    lat_arr, lon_arr,
    input_grid, pred_grid, actual_grid,
    t0_label, forecast_label, interval_minutes,
    out_path=None, interp_factor=24,
):
    """
    Render figure 6-panel modern minimalist dark theme.
    """
    lat_fine, lon_fine, input_fine = interpolate_grid(lat_arr, lon_arr, input_grid, interp_factor)
    _, _, pred_fine = interpolate_grid(lat_arr, lon_arr, pred_grid, interp_factor)

    has_actual = actual_grid is not None and not np.all(np.isnan(actual_grid))
    if has_actual:
        _, _, actual_fine = interpolate_grid(lat_arr, lon_arr, actual_grid, interp_factor)
        mae = float(np.nanmean(np.abs(pred_grid - actual_grid)))
        ctt_min = CTT_VMIN
        ctt_max = CTT_VMAX
        ctt_range = max(ctt_max - ctt_min, 1e-6)
        accuracy = max(0.0, 100.0 * (1 - mae / ctt_range))
    else:
        actual_fine = None
        mae, ctt_min, ctt_max, accuracy = None, None, None, None

    risk_cat_grid = classify(pred_grid, RISK_BINS)
    worst_risk_idx = int(np.nanmax(risk_cat_grid))
    risk_status = RISK_LABELS[worst_risk_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18,9), dpi=180)
    main_axes = list(axes.flat)

    title_line1 = f"Bandung Weather Forecast • {interval_minutes} min"
    if has_actual:
        title_line2 = f"{t0_label} → {forecast_label}  |  {risk_status}  |  MAE: {mae:.2f}K • Acc: {accuracy:.1f}%"
    else:
        title_line2 = f"{t0_label} → {forecast_label}  |  {risk_status}  |  Forecast Mode"

    fig.suptitle(title_line1, color=FG_COLOR, fontsize=15, fontweight="700", y=0.97)
    fig.text(0.5, 0.915, title_line2, color="#38BDF8", fontsize=10.5, ha="center", fontweight="400")

    # Input & Prediksi selalu memakai skala CTT (K) fiks (CTT_VMIN..CTT_VMAX)
    _plot_continuous(axes[0, 0], lat_fine, lon_fine, input_fine, f"Input • {t0_label}")
    _plot_continuous(axes[0, 1], lat_fine, lon_fine, pred_fine, f"Prediksi • {forecast_label}")

    if has_actual:
        _plot_continuous(axes[0, 2], lat_fine, lon_fine, actual_fine, f"Aktual • {forecast_label}")
        _plot_categorical(axes[1, 0], lat_fine, lon_fine, actual_fine, CLOUD_BINS, CLOUD_LABELS, CLOUD_COLORS, "Kelas Awan")

        error_fine = np.abs(pred_fine - actual_fine)
        axes[1, 2].set_title(f"Error Map  (MAE = {mae:.3f}K)", color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
        cf = axes[1, 2].contourf(
            lon_fine, lat_fine, error_fine,
            levels=ERROR_LEVELS, vmin=ERROR_VMIN, vmax=ERROR_VMAX,
            cmap=GLASS_ERROR, alpha=0.96, extend="max",antialiased=True,
        )
        axes[1,2].contour(
            lon_fine,
            lat_fine,
            error_fine,
            levels=ERROR_LEVELS,
            colors="white",
            linewidths=0.15,
            alpha=0.15,
        )
        cb = plt.colorbar(cf, ax=axes[1, 2], fraction=0.046, pad=0.03, shrink=0.83, aspect=22, ticks=ERROR_TICKS)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
        cb.set_label("Error (K)", color=MUTED_COLOR, fontsize=7)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        _style_axis(axes[1, 2], lat_fine, lon_fine)
    else:
        # Panel placeholder tetap menggambar colorbar/legend kosong dengan
        # geometri sama seperti versi berisi data, supaya ukuran panel identik.
        _plot_placeholder(
            axes[0, 2], lat_fine, lon_fine, f"Aktual • {forecast_label}",
            kind="continuous", cmap=GLASS_THERMAL, vmin=CTT_VMIN, vmax=CTT_VMAX,
            ticks=CTT_TICKS, cb_label="CTT (K)",
        )
        _plot_placeholder(
            axes[1, 0], lat_fine, lon_fine, "Kelas Awan",
            kind="categorical", labels=CLOUD_LABELS, colors=CLOUD_COLORS,
        )
        _plot_placeholder(
            axes[1, 2], lat_fine, lon_fine, "Error Map",
            kind="continuous", cmap=GLASS_ERROR, vmin=ERROR_VMIN, vmax=ERROR_VMAX,
            ticks=ERROR_TICKS, cb_label="Error (K)",
        )

    _plot_categorical(
        axes[1, 1], lat_fine, lon_fine, pred_fine, RISK_BINS, RISK_LABELS, RISK_COLORS,
        f"Risk Banjir • {risk_status}",
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.92],
        h_pad=5.5,
        w_pad=1,
    )

    _add_gradient_background(fig)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path
    return fig


def create_forecast_gif(
    frames, lat_arr, lon_arr,
    t0_label="T0", interval_minutes=10,
    output_dir=None, filename="forecast.gif", interp_factor=24, fps=2,
):
    """
    Animasi rollout forecast multi-step (``frames``, shape
    ``(horizon, n_lat, n_lon)``) sebagai GIF. Tiap frame dirender pakai
    style panel kontinu yang sama dengan ``render_six_panel``
    (``_plot_continuous`` + batas Kota Bandung dari ``KotaBandung.geojson``
    lewat ``_style_axis``/``_draw_kecamatan``), jadi tampilannya konsisten
    dengan dashboard.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    _add_gradient_background(fig)

    def _draw(i):
        ax.clear()
        lat_fine, lon_fine, values_fine = interpolate_grid(
            lat_arr, lon_arr, frames[i], interp_factor,
        )
        step_label = f"{t0_label} + {(i + 1) * interval_minutes:.0f} min"
        _plot_continuous(ax, lat_fine, lon_fine, values_fine, f"Forecast • {step_label}")

    anim = FuncAnimation(fig, _draw, frames=len(frames), interval=1000 // fps)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        anim.save(out_path, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": GRAD_TOP})
        plt.close(fig)
        return out_path

    plt.close(fig)
    return anim


def plot_confusion_matrix(confusion, labels, output_dir=None, filename="confusion_matrix.png"):
    """
    Confusion matrix dengan tema dark yang senada dengan
    ``render_six_panel`` (warna & font sama), tanpa overlay peta karena
    ini bukan visualisasi spasial.
    """
    confusion = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=180)
    _add_gradient_background(fig)
    ax.set_facecolor("none")

    cmap = LinearSegmentedColormap.from_list("cm_glass", [CARD_FACE, "#1D66B8", "#52A6FF"])
    im = ax.imshow(confusion, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=MUTED_COLOR, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(labels, color=MUTED_COLOR, fontsize=8)
    ax.set_xlabel("Prediksi", color=FG_COLOR, fontsize=9, fontweight="700")
    ax.set_ylabel("Aktual", color=FG_COLOR, fontsize=9, fontweight="700")
    ax.set_title("Confusion Matrix", color=FG_COLOR, fontsize=12, fontweight="700", pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    vmax = confusion.max() if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            text_color = FG_COLOR if value < vmax * 0.6 else GRAD_TOP
            ax.text(
                j, i, f"{value:.0f}",
                ha="center", va="center", color=text_color, fontsize=9, fontweight="700",
            )

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.83)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)

    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path

    plt.close(fig)
    return fig


def create_dashboard_gif(
    input_grid, pred_frames, lat_arr, lon_arr,
    t0_label, interval_minutes,
    actual_frames=None,
    output_dir=None, filename="dashboard.gif",
    interp_factor=24, fps=1,
):
    """
    Animasi GIF dari dashboard 6-panel lengkap: tiap frame adalah satu
    langkah forecast ke depan, dan tiap frame dirender dengan memanggil
    ``render_six_panel`` yang sama persis (bukan disalin ulang), supaya
    stylingnya selalu identik dengan dashboard statis. Frame digabung
    jadi GIF beneran (banyak frame berbeda) lewat Pillow, bukan lewat
    ``matplotlib.animation`` -- lebih aman untuk figure kompleks 2x3
    subplot seperti ini.

    Args:
        input_grid: Grid CTT observasi di t0, shape ``(n_lat, n_lon)``.
            Panel "Input" tetap sama di semua frame (titik awal rollout).
        pred_frames: Array prediksi tiap langkah, shape
            ``(n_steps, n_lat, n_lon)`` (mis. ``frames`` dari
            ``forecasting.bias_correction``, di-reshape).
        lat_arr, lon_arr: Koordinat asli grid (lihat ``interpolate_grid``).
        t0_label: Label t0 untuk judul (lihat ``render_six_panel``).
        interval_minutes: Menit riil antar langkah, dipakai untuk label
            tiap frame (``t0_label + i * interval_minutes``) dan judul.
        actual_frames: Opsional, array data aktual per langkah dengan
            shape sama seperti ``pred_frames``, untuk backtest (kalau
            ada). ``None`` (default) berarti setiap frame menampilkan
            placeholder "Data aktual belum tersedia", sama seperti
            dashboard statis saat horizon melewati data yang tersedia.
        output_dir, filename: Kalau ``output_dir`` diisi, GIF disimpan
            ke ``output_dir / filename`` dan path itu yang dikembalikan;
            kalau tidak, mengembalikan list frame PIL.Image mentah.
        interp_factor: Diteruskan ke ``render_six_panel``.
        fps: Frame per detik animasi.

    Returns:
        Path ke file GIF (kalau ``output_dir`` diisi) atau list
        ``PIL.Image`` (kalau tidak).
    """
    import io
    from PIL import Image

    pil_frames = []
    for i in range(len(pred_frames)):
        forecast_label = f"{t0_label} + {(i + 1) * interval_minutes:.0f} menit"
        actual_grid = actual_frames[i] if actual_frames is not None else None
        fig = render_six_panel(
            lat_arr, lon_arr,
            input_grid, pred_frames[i], actual_grid,
            t0_label, forecast_label, interval_minutes,
            out_path=None, interp_factor=interp_factor,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, facecolor=GRAD_TOP)
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("RGB"))
        buf.close()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        pil_frames[0].save(
            out_path, save_all=True, append_images=pil_frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        return out_path

    return pil_frames


def render_interval_comparison(results, output_dir=None, filename="interval_comparison.png"):
    """
    Ringkasan bar chart MAE & akurasi untuk beberapa skenario interval
    forecast menuju horizon target yang sama (mis. 10/30/60 menit
    menuju +3 jam), tema dark senada dashboard, tanpa peta. MAE/akurasi
    dihitung di sini dengan formula yang sama seperti ``render_six_panel``,
    supaya konsisten satu sumber kebenaran.

    Args:
        results: List of dict, tiap dict:
            ``{"interval_minutes": int, "pred_grid": array 2D,
            "actual_grid": array 2D atau None}``. Kalau
            ``actual_grid`` ``None`` (tidak ada data aktual buat
            horizon itu), MAE/akurasi ditampilkan "N/A" -- tidak
            dikarang.
        output_dir, filename: Kalau ``output_dir`` diisi, figure
            disimpan ke ``output_dir / filename`` dan path itu yang
            dikembalikan; kalau tidak, mengembalikan figure.
    """
    labels, mae_vals, acc_vals = [], [], []
    for r in results:
        labels.append(f"{r['interval_minutes']} menit")
        actual_grid = r.get("actual_grid")
        if actual_grid is not None and not np.all(np.isnan(actual_grid)):
            mae = float(np.nanmean(np.abs(r["pred_grid"] - actual_grid)))
            ctt_range = max(CTT_VMAX - CTT_VMIN, 1e-6)
            accuracy = max(0.0, 100.0 * (1 - mae / ctt_range))
        else:
            mae, accuracy = None, None
        mae_vals.append(mae)
        acc_vals.append(accuracy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
    _add_gradient_background(fig)
    bar_colors = ["#52A6FF", "#F6B532", "#EB521C"][: len(results)]

    for ax, values, title, ylabel, fmt in (
        (ax1, mae_vals, "MAE per Interval", "MAE (K)", "{:.2f}"),
        (ax2, acc_vals, "Akurasi per Interval", "Akurasi (%)", "{:.1f}%"),
    ):
        ax.set_facecolor("none")
        plot_values = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(labels, plot_values, color=bar_colors, width=0.55, edgecolor="none")
        ax.set_title(title, color=FG_COLOR, fontsize=12, fontweight="700", pad=10)
        ax.set_ylabel(ylabel, color=MUTED_COLOR, fontsize=9)
        ax.tick_params(colors=MUTED_COLOR, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=CARD_EDGE, alpha=0.3, linewidth=0.6)
        for bar, value in zip(bars, values):
            label_text = fmt.format(value) if value is not None else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label_text, ha="center", va="bottom",
                color=FG_COLOR, fontsize=9, fontweight="700",
            )

    fig.suptitle("Perbandingan Interval Forecast (asumsi cadence per langkah)", color=FG_COLOR, fontsize=14, fontweight="700", y=1.02)
    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path

    plt.close(fig)
    return fig


def render_window_size_search(payload, output_dir=None, filename="window_size_search.png"):
    """
    Ringkasan hasil window-size search (``experiment.run_window_size_search``'s
    ``{"results": [...], "best": {...}}``) sebagai chart: RMSE & MAE
    (grouped bar) dan R^2 per window_size, dengan window_size terbaik
    ditandai beda (warna + bintang), tema senada dashboard lain.

    Args:
        payload: Dict dengan keys ``"results"`` (list of dict, tiap dict
            minimal punya ``window_size``, ``rmse``, ``mae``, ``r2``)
            dan ``"best"`` (satu dict dari ``results`` yang sama, dengan
            ``window_size`` yang menandai kandidat terbaik).
        output_dir, filename: Kalau ``output_dir`` diisi, figure
            disimpan ke ``output_dir / filename`` dan path itu yang
            dikembalikan; kalau tidak, mengembalikan figure.
    """
    results = sorted(payload["results"], key=lambda r: r["window_size"])
    best_window_size = payload["best"]["window_size"]

    labels = [str(r["window_size"]) for r in results]
    rmse_vals = [r["rmse"] for r in results]
    mae_vals = [r["mae"] for r in results]
    r2_vals = [r["r2"] for r in results]
    is_best = [r["window_size"] == best_window_size for r in results]

    best_color, normal_color = "#52A6FF", "#35516B"
    bar_colors = [best_color if b else normal_color for b in is_best]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    _add_gradient_background(fig)

    # ---- RMSE & MAE (grouped bars) ----
    ax1.set_facecolor("none")
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, rmse_vals, width, color=bar_colors, edgecolor="none", label="RMSE")
    ax1.bar(x + width / 2, mae_vals, width, color=bar_colors, alpha=0.55, edgecolor="none", label="MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color=MUTED_COLOR, fontsize=9)
    ax1.set_title("RMSE & MAE per Window Size", color=FG_COLOR, fontsize=12, fontweight="700", pad=10)
    ax1.set_xlabel("Window Size", color=MUTED_COLOR, fontsize=9)
    ax1.tick_params(colors=MUTED_COLOR, labelsize=9, length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.grid(axis="y", color=CARD_EDGE, alpha=0.3, linewidth=0.6)
    legend1 = ax1.legend(facecolor=CARD_FACE, edgecolor="none", labelcolor=FG_COLOR, fontsize=8, loc="upper right")
    legend1.get_frame().set_alpha(0.6)

    # ---- R^2 ----
    ax2.set_facecolor("none")
    bars2 = ax2.bar(labels, r2_vals, color=bar_colors, width=0.55, edgecolor="none")
    ax2.set_title("R\u00b2 per Window Size", color=FG_COLOR, fontsize=12, fontweight="700", pad=10)
    ax2.set_xlabel("Window Size", color=MUTED_COLOR, fontsize=9)
    ax2.tick_params(colors=MUTED_COLOR, labelsize=9, length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(axis="y", color=CARD_EDGE, alpha=0.3, linewidth=0.6)
    for bar, value, best in zip(bars2, r2_vals, is_best):
        star_prefix = "\u2605 " if best else ""
        label_text = f"{star_prefix}{value:.3f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            label_text, ha="center", va="bottom",
            color=FG_COLOR, fontsize=9, fontweight="700",
        )

    fig.suptitle(
        f"Window Size Search \u2014 Terbaik: {best_window_size} "
        f"(RMSE={payload['best']['rmse']:.3f}, MAE={payload['best']['mae']:.3f}, "
        f"R\u00b2={payload['best']['r2']:.3f})",
        color=FG_COLOR, fontsize=13, fontweight="700", y=1.03,
    )
    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path

    plt.close(fig)
    return fig


def render_cross_validation(cv_result, output_dir=None, filename="cross_validation.png"):
    """
    Ringkasan hasil cross-validation (``experiment.run_cross_validation``'s
    ``{"window_size", "model", "n_splits", "folds": [...], "summary": {...}}``)
    sebagai chart: tiap fold digambar sebagai titik terpisah (RMSE & MAE),
    plus garis rata-rata dan pita +/-1 std, supaya sebaran antar fold
    kelihatan -- bukan cuma satu angka rata-rata. Tema senada dashboard
    lain.

    Args:
        cv_result: Dict dari ``experiment.run_cross_validation``, minimal
            punya ``"folds"`` (list of dict, tiap dict punya ``fold``,
            ``rmse``, ``mae``, ``r2``) dan ``"summary"`` (dict
            ``{"rmse": {"mean","std"}, "mae": {...}, "r2": {...}}``).
        output_dir, filename: Kalau ``output_dir`` diisi, figure
            disimpan ke ``output_dir / filename`` dan path itu yang
            dikembalikan; kalau tidak, mengembalikan figure.
    """
    folds = sorted(cv_result["folds"], key=lambda r: r["fold"])
    summary = cv_result["summary"]
    fold_labels = [str(r["fold"] + 1) for r in folds]
    x = np.arange(len(folds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    _add_gradient_background(fig)

    for ax, key, title, color in (
        (ax1, "rmse", "RMSE per Fold", "#52A6FF"),
        (ax2, "mae", "MAE per Fold", "#F6B532"),
    ):
        values = [r[key] for r in folds]
        mean, std = summary[key]["mean"], summary[key]["std"]

        ax.set_facecolor("none")
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.15, zorder=1)
        ax.axhline(mean, color=color, linewidth=1.4, linestyle="--", zorder=2)
        ax.scatter(x, values, color=color, s=70, zorder=3, edgecolor=GRAD_TOP, linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {lbl}" for lbl in fold_labels], color=MUTED_COLOR, fontsize=9)
        ax.set_title(title, color=FG_COLOR, fontsize=12, fontweight="700", pad=10)
        ax.tick_params(colors=MUTED_COLOR, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=CARD_EDGE, alpha=0.3, linewidth=0.6)

        ax.text(
            0.02, 0.96, f"rata-rata = {mean:.4f} \u00b1 {std:.4f}",
            transform=ax.transAxes, ha="left", va="top",
            color=FG_COLOR, fontsize=9, fontweight="700",
        )

    fig.suptitle(
        f"Cross-Validation ({cv_result['n_splits']}-fold) \u2014 "
        f"{cv_result['model']}, window={cv_result['window_size']} \u2014 "
        f"R\u00b2={summary['r2']['mean']:.3f}\u00b1{summary['r2']['std']:.3f}",
        color=FG_COLOR, fontsize=13, fontweight="700", y=1.03,
    )
    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path

    plt.close(fig)
    return fig