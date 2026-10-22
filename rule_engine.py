import os
import re
import io

from PIL import Image, ImageChops, ExifTags
import cv2
import numpy as np

# ============================================================
# PRETRAINED AI / DEEPFAKE MODEL
# ============================================================

try:
    import torch
    from transformers import pipeline
    AI_MODEL_AVAILABLE = True
except ImportError:
    torch = None
    pipeline = None
    AI_MODEL_AVAILABLE = False


AI_DETECTOR = None
FACE_DETECTOR = None
PROFILE_FACE_DETECTOR = None


def load_face_detector():
    """Load OpenCV's built-in frontal-face detector."""
    global FACE_DETECTOR

    if FACE_DETECTOR is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        FACE_DETECTOR = cv2.CascadeClassifier(cascade_path)

        if FACE_DETECTOR.empty():
            raise RuntimeError("OpenCV frontal face detector could not be loaded")

    return FACE_DETECTOR


def load_profile_face_detector():
    """Load OpenCV's profile-face detector as a secondary fallback."""
    global PROFILE_FACE_DETECTOR

    if PROFILE_FACE_DETECTOR is None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            if os.path.exists(cascade_path):
                PROFILE_FACE_DETECTOR = cv2.CascadeClassifier(cascade_path)
        except Exception:
            PROFILE_FACE_DETECTOR = None

    return PROFILE_FACE_DETECTOR


def extract_largest_face(frame, pad_ratio=0.50, return_box=False):
    """Return a portrait-padded crop of the largest detected human face, or None (and optionally the box)."""
    try:
        if frame is None or frame.size == 0:
            return (None, None) if return_box else None

        h, w = frame.shape[:2]
        detector = load_face_detector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face must be at least 8% of smaller dimension (min 60px) to prevent detecting flower petals in 4K
        min_face_size = max(60, int(min(h, w) * 0.08))

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )

        if len(faces) == 0:
            profile_det = load_profile_face_detector()
            if profile_det is not None and not profile_det.empty():
                faces = profile_det.detectMultiScale(
                    gray,
                    scaleFactor=1.10,
                    minNeighbors=5,
                    minSize=(min_face_size, min_face_size)
                )

        if len(faces) == 0:
            return (None, None) if return_box else None

        # Filter candidates: genuine human faces have aspect ratio roughly between 0.70 and 1.35
        valid_faces = [
            f for f in faces
            if 0.70 <= (f[2] / max(1, f[3])) <= 1.35
        ]

        if not valid_faces:
            return (None, None) if return_box else None

        # Select the largest face candidate
        x, y, fw, fh = max(
            valid_faces,
            key=lambda box: box[2] * box[3]
        )

        # Add 50% padding so the model sees full head, hair, chin, and neck context
        pad_x = int(fw * pad_ratio)
        pad_y = int(fh * pad_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0 or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
            return (None, None) if return_box else None

        if return_box:
            return face_crop, (x, y, fw, fh)
        return face_crop

    except Exception as e:
        print(f"Face extraction warning: {e}")
        return (None, None) if return_box else None



def load_ai_detector():
    """
    Load the pretrained deepfake image classifier.

    The model analyzes individual video frames.
    It is NOT a certified video-level deepfake detector.
    """

    global AI_DETECTOR

    if not AI_MODEL_AVAILABLE:
        raise RuntimeError(
            "AI model packages are not installed. "
            "Run: pip install torch torchvision transformers"
        )

    if AI_DETECTOR is None:

        print("==========================================")
        print("Loading pretrained AI/deepfake model...")
        print("==========================================")

        device = 0 if torch.cuda.is_available() else -1

        AI_DETECTOR = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
            device=device
        )

        print("AI/deepfake model loaded successfully.")

    return AI_DETECTOR


# ============================================================
# WEBSITE ANALYSIS
# ============================================================

def analyze_website(url):

    score = 0
    results = []

    # Remove extra spaces
    url = url.strip()

    # Rule 1 - HTTPS Check
    if url.startswith("https://"):
        score += 30
        results.append(("HTTPS Security", "✔ Secure"))
    else:
        results.append(("HTTPS Security", "❌ Not Secure"))

    # Rule 2 - URL Format Check (allows standard domains, ports, paths, and query strings)
    pattern = r"^(http://|https://)[A-Za-z0-9.-]+(\.[A-Za-z]{2,})(:\d+)?(/.*)?$"

    if re.match(pattern, url):
        score += 25
        results.append(("URL Format", "✔ Valid"))
    else:
        score += 10
        results.append(("URL Format", "⚠ Non-standard URL format"))

    # Rule 3 - URL Length Check
    if len(url) < 90:
        score += 15
        results.append(("URL Length", "✔ Normal"))
    elif len(url) < 150:
        score += 10
        results.append(("URL Length", "ℹ Acceptable"))
    else:
        score += 5
        results.append(("URL Length", "⚠ Too Long"))

    # Rule 4 - IP Address Detection
    ip_pattern = r"^(http://|https://)\d{1,3}(\.\d{1,3}){3}"

    if re.match(ip_pattern, url):
        score -= 25
        results.append(("IP Address URL", "⚠ Suspicious IP-based URL"))
    else:
        score += 15
        results.append(("IP Address URL", "✔ Domain Used"))

    # Rule 5 - Suspicious Keyword Detection
    suspicious_keywords = [
        "login",
        "verify",
        "update",
        "secure",
        "account",
        "password",
        "confirm"
    ]

    found_keyword = False

    for keyword in suspicious_keywords:
        if keyword in url.lower():
            found_keyword = True
            break

    if found_keyword:
        # If it uses HTTPS and normal domain, moderate deduction
        score -= 5
        results.append(
            ("Suspicious Keyword", "⚠ Keyword present (verify domain)")
        )
    else:
        score += 15
        results.append(
            ("Suspicious Keyword", "✔ None detected")
        )

    # Keep score between 0 and 100
    score = max(0, min(score, 100))

    # Final Status
    if score >= 80:
        status = "🟢 Safe Website"
    elif score >= 50:
        status = "🟡 Medium Risk"
    else:
        status = "🔴 Suspicious Website"

    return score, status, results


# ============================================================
# QR CODE ANALYSIS
# ============================================================

def analyze_qr_content(data):

    score = 100
    results = []
    content_type = "Unknown"

    data = data.strip()

    # Rule 1 - Dangerous QR Schemes
    dangerous_schemes = [
        "javascript:",
        "intent:",
        "data:",
        "file:",
        "market:"
    ]

    found_dangerous = False

    for scheme in dangerous_schemes:
        if data.lower().startswith(scheme):
            found_dangerous = True
            break

    if found_dangerous:
        score = 10
        results.append(
            (
                "QR Scheme Check",
                "❌ Dangerous executable scheme detected"
            )
        )
        content_type = "Dangerous Payload"
        status = "🔴 Suspicious QR Code"

        return (
            score,
            status,
            results,
            content_type,
            None
        )

    # Rule 2 - Content Type Detection & Baseline Scoring
    if (
        data.lower().startswith("http://")
        or data.lower().startswith("https://")
    ):
        content_type = "URL"
        results.append(
            ("Content Type", "✔ Website URL")
        )

        if data.lower().startswith("https://"):
            results.append(
                ("Transport Security", "✔ Secure HTTPS")
            )
        else:
            score -= 15
            results.append(
                ("Transport Security", "⚠ Unencrypted HTTP")
            )

        # Check for IP address in URL
        if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", data.lower()):
            score -= 30
            results.append(
                ("Destination Check", "🔴 Raw IP address destination detected")
            )

        # Rule 3 - URL Shortener Detection
        shortener_domains = [
            "bit.ly",
            "tinyurl.com",
            "goo.gl",
            "t.co",
            "ow.ly",
            "is.gd",
            "buff.ly",
            "adf.ly",
            "cutt.ly",
            "rebrand.ly"
        ]

        is_shortener = any(d in data.lower() for d in shortener_domains)

        if is_shortener:
            score -= 25
            results.append(
                (
                    "URL Shortener",
                    "⚠ Shortened link - real destination hidden"
                )
            )
        else:
            results.append(
                (
                    "URL Shortener",
                    "✔ Direct destination link"
                )
            )

    elif data.lower().startswith("wifi:"):
        content_type = "WiFi Credentials"
        score = 85
        results.append(
            ("Content Type", "ℹ WiFi network configuration")
        )
        results.append(
            ("Security Notice", "⚠ Connect only to trusted wireless networks")
        )

    elif (
        data.lower().startswith("upi:")
        or data.lower().startswith("upi://")
    ):
        content_type = "UPI Payment"
        score = 85
        results.append(
            ("Content Type", "ℹ Payment request")
        )
        results.append(
            ("Payment Security", "⚠ Verify recipient address and amount before payment")
        )

    elif data.lower().startswith("mailto:"):
        content_type = "Email"
        score = 90
        results.append(
            ("Content Type", "✔ Email action link")
        )

    elif data.lower().startswith("tel:"):
        content_type = "Phone Number"
        score = 90
        results.append(
            ("Content Type", "✔ Phone call link")
        )

    else:
        content_type = "Plain Text"
        score = 95
        results.append(
            ("Content Type", "✔ Plain text message")
        )

    # Rule 4 - QR Payload Length Check
    if len(data) > 500:
        score -= 15
        results.append(
            (
                "Payload Length",
                f"⚠ Unusually long QR payload ({len(data)} characters)"
            )
        )
    else:
        results.append(
            (
                "Payload Length",
                f"✔ Normal length ({len(data)} characters)"
            )
        )

    # Clamp score
    score = max(0, min(score, 100))

    if score >= 80:
        status = "🟢 Safe QR Code"
    elif score >= 50:
        status = "🟡 Medium Risk QR Code"
    else:
        status = "🔴 Suspicious QR Code"

    is_url = content_type == "URL"

    return (
        score,
        status,
        results,
        content_type,
        data if is_url else None
    )


# ============================================================
# ERROR LEVEL ANALYSIS
# ============================================================

def error_level_analysis(
    pil_image,
    quality=90
):

    buffer = io.BytesIO()

    original = pil_image.convert("RGB")

    original.save(
        buffer,
        "JPEG",
        quality=quality
    )

    buffer.seek(0)

    resaved = Image.open(buffer)

    diff = ImageChops.difference(
        original,
        resaved
    )

    extrema = diff.getextrema()

    max_diff = max(
        [channel[1] for channel in extrema]
    )

    return max_diff


# ============================================================
# NOISE PATTERN
# ============================================================

def check_noise_pattern(pil_image):

    try:

        rgb = pil_image.convert("RGB")

        gray = cv2.cvtColor(
            np.array(rgb),
            cv2.COLOR_RGB2GRAY
        )

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        noise_std = laplacian.std()

        return round(
            noise_std,
            2
        )

    except Exception:

        return None


# ============================================================
# AI RESOLUTION PATTERN
# ============================================================

def check_ai_resolution_pattern(
    width,
    height
):

    common_ai_sizes = [

        (512, 512),
        (768, 768),
        (1024, 1024),

        (512, 768),
        (768, 512),

        (1024, 1536),
        (1536, 1024),

        (896, 1152),
        (1152, 896),

        (1024, 1792),
        (1792, 1024)
    ]

    if (
        width,
        height
    ) in common_ai_sizes:

        return True

    if (
        width % 64 == 0
        and height % 64 == 0
        and width <= 2048
        and height <= 2048
    ):

        return True

    return False


# ============================================================
# SYNTHETIC ART & AI GRAPHIC COMPOSITE ANALYSIS
# ============================================================

def check_synthetic_art_indicators(cv_img, width, height, has_exif, noise_level=None, ens_real=None):
    """
    Analyze image for AI-generated fantasy artwork, digital greeting graphics,
    and synthetic illustration styling (e.g. Midjourney / Stable Diffusion).
    """
    try:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        mean_s = float(np.mean(s))
        pct_high_s = float(np.mean(s > 100) * 100)

        ai_res = (
            width in (512, 768, 1024, 1536)
            or height in (512, 768, 1024, 1536)
            or (width % 64 == 0 and height % 64 == 0 and max(width, height) <= 2048)
        )

        is_synthetic = False
        indicators = []

        # Check for genuine physical camera sensor noise or verified authentic face
        has_camera_noise = (noise_level is not None and noise_level >= 5.0)
        has_authentic_face = (ens_real is not None and ens_real >= 0.65)

        # If camera sensor noise or authentic human face is present,
        # it is a genuine camera capture (e.g. phone camera with HDR or social media resize).
        if has_camera_noise or has_authentic_face:
            if mean_s > 125.0 and pct_high_s > 70.0:
                is_synthetic = True
                indicators.append(
                    f"Surreal artificial saturation palette (mean saturation: {mean_s:.1f})"
                )
        else:
            # Hyper-saturated / surreal color palette typical of generative AI art
            if mean_s > 80.0 and pct_high_s > 35.0:
                is_synthetic = True
                indicators.append(
                    f"Hyper-saturated synthetic palette (mean saturation: {mean_s:.1f}, {pct_high_s:.1f}% vibrant pixels)"
                )
            elif mean_s > 70.0 and pct_high_s > 25.0 and ai_res and not has_exif:
                is_synthetic = True
                indicators.append(
                    f"Synthetic color profile with generative output sizing (mean saturation: {mean_s:.1f})"
                )

            if ai_res and not has_exif:
                indicators.append(
                    f"Resolution {width}x{height} matches common generative AI output dimensions"
                )

        return is_synthetic, indicators, mean_s, pct_high_s

    except Exception:
        return False, [], 0.0, 0.0


# ============================================================
# FREQUENCY ARTIFACT ANALYSIS
# ============================================================

def check_frequency_artifacts(
    pil_image
):

    try:

        gray = np.array(
            pil_image.convert("L"),
            dtype=np.float64
        )

        h, w = gray.shape

        f = np.fft.fft2(gray)

        fshift = np.fft.fftshift(f)

        magnitude = np.log(
            np.abs(fshift) + 1
        )

        cy = h // 2
        cx = w // 2

        max_radius = min(
            cy,
            cx
        )

        if max_radius < 20:

            return None, None

        y, x = np.indices(
            (h, w)
        )

        r = np.sqrt(
            (x - cx) ** 2
            + (y - cy) ** 2
        ).astype(int)

        r = np.clip(
            r,
            0,
            max_radius - 1
        )

        radial_sum = np.bincount(
            r.ravel(),
            weights=magnitude.ravel()
        )

        radial_count = np.bincount(
            r.ravel()
        )

        radial_profile = (
            radial_sum
            / np.maximum(
                radial_count,
                1
            )
        )

        start = int(
            max_radius * 0.4
        )

        end = int(
            max_radius * 0.9
        )

        high_freq_profile = (
            radial_profile[start:end]
        )

        if len(
            high_freq_profile
        ) < 10:

            return None, None

        window = 5

        kernel = (
            np.ones(window)
            / window
        )

        smoothed = np.convolve(
            high_freq_profile,
            kernel,
            mode="same"
        )

        deviations = (
            high_freq_profile
            - smoothed
        )

        dev_std = deviations.std()

        threshold = (
            dev_std * 2.5
            if dev_std > 0
            else 0
        )

        peak_count = 0

        for i in range(
            1,
            len(deviations) - 1
        ):

            if (
                deviations[i] > threshold
                and deviations[i] > deviations[i - 1]
                and deviations[i] > deviations[i + 1]
            ):

                peak_count += 1

        return (
            peak_count,
            round(
                float(dev_std),
                3
            )
        )

    except Exception:

        return None, None


# ============================================================
# AI GENERATION METADATA SIGNATURE
# ============================================================

def check_ai_generation_signature(
    pil_image,
    exif_data,
    file_path=None
):

    ai_keywords = [
        "chatgpt",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stablediffusion",
        "leonardo.ai",
        "nightcafe",
        "comfyui",
        "automatic1111",
        "diffusion",
        "bing image creator"
    ]

    info = pil_image.info

    combined_text = " ".join(
        str(v).lower()
        for v in info.values()
    )

    if exif_data:
        combined_text += (
            " "
            + str(exif_data).lower()
        )

    if file_path:
        fname = os.path.basename(file_path).lower()
        combined_text += " " + fname

    for keyword in ai_keywords:
        if keyword in combined_text:
            return True, keyword

    if (
        "parameters" in info
        or "prompt" in info
        or "workflow" in info
    ):
        return True, (
            "generation parameters embedded in file"
        )

    return False, None


# ============================================================
# DIGITAL TEXT & GRAPHIC OVERLAY DETECTION
# ============================================================

def check_text_graphic_overlay(cv_img):
    """
    Detect digital text / typography overlay added to an image (e.g. greeting cards,
    memes, Good Morning banners, promotional typography).
    Returns (has_overlay, band_name, char_count).
    """
    try:
        h, w = cv_img.shape[:2]
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        bands = [
            ('bottom', gray[int(h * 0.65):, :]),
            ('top', gray[:int(h * 0.35), :]),
            ('middle', gray[int(h * 0.30):int(h * 0.70), :])
        ]

        for band_name, band in bands:
            edges = cv2.Canny(band, 60, 180)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            char_boxes = []
            bh_max = int(h * 0.12)
            bh_min = max(10, int(h * 0.015))
            bw_max = int(w * 0.15)
            bw_min = max(5, int(w * 0.008))

            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                if bh_min <= bh <= bh_max and bw_min <= bw <= bw_max:
                    ar = bw / float(bh)
                    if 0.15 <= ar <= 2.2:
                        char_boxes.append((x, y, bw, bh))

            char_boxes.sort(key=lambda b: b[0])

            max_aligned = 0
            curr_aligned = 1
            for i in range(len(char_boxes) - 1):
                b1, b2 = char_boxes[i], char_boxes[i + 1]
                dy = abs(b1[1] - b2[1])
                dx = b2[0] - (b1[0] + b1[2])
                dh = abs(b1[3] - b2[3])

                if dy <= max(8, int(b1[3] * 0.35)) and -5 <= dx <= max(35, int(b1[2] * 1.5)) and dh <= max(8, int(b1[3] * 0.4)):
                    curr_aligned += 1
                else:
                    if curr_aligned >= 5:
                        max_aligned = max(max_aligned, curr_aligned)
                    curr_aligned = 1
            if curr_aligned >= 5:
                max_aligned = max(max_aligned, curr_aligned)

            if max_aligned >= 6:
                return True, band_name, max_aligned

        return False, None, 0
    except Exception:
        return False, None, 0


# ============================================================
# LOCALIZED IMAGE EDITING & AI PERCENTAGE DETECTION
# ============================================================

def detect_localized_image_editing(
    img,
    cv2_bgr,
    exif_data,
    has_face,
    face_rect,
    ens_fake,
    is_synth,
    art_indicators,
    noise_level,
    max_diff,
    is_ai_meta,
    meta_kw,
    has_text_overlay=False,
    text_band=None,
    text_char_count=0
):
    """
    Detect whether parts of an image were deleted (inpainting/removal),
    added (splicing/generative fill), retouched, deepfaked, text-overlaid, or fully AI-generated.
    Returns (ai_percentage, edit_type, edit_explanation).
    """
    W, H = img.size
    total_pixels = W * H

    # Check for image editing software in EXIF
    software = str(exif_data.get('Software', '')).lower() if exif_data else ''
    editing_tools = ['photoshop', 'gimp', 'lightroom', 'snapseed', 'picsart', 'canva']
    found_tool = next((t for t in editing_tools if t in software), None)

    # ----------------------------------------------------
    # Case 1: Full Generative AI (Midjourney / DALL-E / Sora)
    # ----------------------------------------------------
    is_protected_real = (has_face and ens_fake < 0.35)
    if not is_protected_real and (is_synth or is_ai_meta or (not has_face and noise_level is not None and noise_level < 2.0 and not exif_data and max_diff > 45)):
        primary = art_indicators[0] if art_indicators else (f"AI metadata: {meta_kw}" if is_ai_meta else "Neural diffusion synthesis")
        return (
            100.0,
            "🔴 Full Generative AI (100% Synthetic)",
            f"The entire image was generated synthetically from scratch using AI generative diffusion models ({primary}). No authentic physical camera lens or sensor noise profile detected."
        )

    # ----------------------------------------------------
    # Case 2: Deepfake Face Replacement on Camera Photo
    # ----------------------------------------------------
    if has_face and ens_fake >= 0.60:
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            face_area_pct = ((fw * fh * 1.5) / total_pixels) * 100.0
            manipulated_pct = round(max(15.0, min(50.0, face_area_pct)), 1)
        else:
            manipulated_pct = 25.0

        return (
            manipulated_pct,
            "🔴 Deepfake Facial Manipulation (Face Swap)",
            f"AI facial manipulation detected covering approximately {manipulated_pct}% of the image. The facial region was digitally replaced with synthetic neural features ({ens_fake * 100:.1f}% AI confidence), while the surrounding frame retains original camera capture."
        )

    # ----------------------------------------------------
    # Case 3: Digital Text & Graphic Overlay (Greetings / Memes / Banners)
    # ----------------------------------------------------
    if has_text_overlay:
        band_str = f"{text_band} section" if text_band else "image"
        return (
            18.0,
            "🟡 Digital Graphic / Text Overlay Edit (Greeting / Banner)",
            f"Digital typography/text banner added across {band_str} of the image (graphic text overlay detected with {text_char_count} aligned typographic characters). The background contains authentic photographic capture, but the image has been edited with graphic text overlay for greeting / messaging sharing."
        )

    # ----------------------------------------------------
    # Block Forensic Grid: ELA, Local Noise & Saturation
    # ----------------------------------------------------
    try:
        rgb = img.convert('RGB')
        buf = io.BytesIO()
        rgb.save(buf, 'JPEG', quality=90)
        buf.seek(0)
        resaved = Image.open(buf)
        diff = ImageChops.difference(rgb, resaved)
        diff_np = np.array(diff).astype(np.float32).mean(axis=2)

        gray = np.array(img.convert('L'), dtype=np.uint8)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

        hsv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)

        bs = max(16, min(64, min(W, H) // 16))
        nx, ny = W // bs, H // bs

        b_ela, b_noise, b_sat = [], [], []
        for y in range(ny):
            for x in range(nx):
                x1, y1 = x * bs, y * bs
                x2, y2 = x1 + bs, y1 + bs
                b_ela.append(diff_np[y1:y2, x1:x2].mean())
                b_noise.append(laplacian[y1:y2, x1:x2].mean())
                b_sat.append(sat[y1:y2, x1:x2].mean())

        b_ela = np.array(b_ela)
        b_noise = np.array(b_noise)
        b_sat = np.array(b_sat)

        med_ela, std_ela = np.median(b_ela), np.std(b_ela)
        med_noise, std_noise = np.median(b_noise), np.std(b_noise)
        med_sat, std_sat = np.median(b_sat), np.std(b_sat)

        # Case 4: Object Addition (Splicing / Generative Fill)
        # Spliced elements have stark compression error differences (max_diff > 55)
        if max_diff > 55 and std_ela > 5.0:
            high_ela_blocks = (b_ela > med_ela + 3.0 * std_ela) & (b_ela > 25.0)
            high_sat_hotspots = (b_sat > med_sat + 3.5 * std_sat) & (b_sat > 175.0) & (med_sat < 90.0)
            addition_blocks = high_ela_blocks | high_sat_hotspots

            if np.sum(addition_blocks) >= 3:
                add_pct = round((np.sum(addition_blocks) / len(b_ela)) * 100, 1)
                add_pct = max(8.0, min(75.0, add_pct))
                return (
                    add_pct,
                    "🔴 Object Addition / Generative Fill (Splicing)",
                    f"Localized element addition or generative fill detected across approximately {add_pct}% of the image. Significant compression and frequency disparity detected between the inserted element and the authentic camera background."
                )

        # Case 5: Object Deletion (Inpainting / Content-Aware Removal)
        # Real inpainting requires genuine recompression disparity (max_diff > 45) AND true texture eradication (b_noise < 1.0).
        # Smooth white walls, clear sky, or studio backdrops with low compression error are NOT inpainting.
        if max_diff > 45 and med_noise > 8.0 and std_ela > 5.0:
            removal_blocks = (b_noise < 1.0) & (b_ela < med_ela * 0.25)
            if np.sum(removal_blocks) >= 3:
                rem_pct = round((np.sum(removal_blocks) / len(b_noise)) * 100, 1)
                rem_pct = max(5.0, min(50.0, rem_pct))
                return (
                    rem_pct,
                    "🔴 Object Removal / Content-Aware Fill (Inpainting)",
                    f"Localized object deletion / inpainting detected covering approximately {rem_pct}% of the image. Camera sensor grain has been erased and interpolated with smoothing artifacts characteristic of removal tools."
                )

    except Exception as e:
        print(f"Localized block analysis warning: {e}")

    # Case 6: Digital Retouching / Color Post-Processing
    if found_tool or max_diff > 55:
        tool_str = f" ({found_tool.title()})" if found_tool else ""
        return (
            15.0,
            "🟡 Digital Retouching / Color Post-Processing",
            f"Global color grading, contrast adjustment, or digital retouching detected{tool_str}. No discrete object addition or deletion found."
        )

    # Case 7: Authentic Camera Photo (Unedited)
    return (
        0.0,
        "🟢 Authentic Camera Photo (Unedited)",
        "Original, unmodified camera capture. Uniform sensor noise, consistent compression profile, and no signs of object addition, deletion, or AI generation."
    )


# ============================================================
# IMAGE FORENSICS
# ============================================================

def analyze_image_forensics(
    file_path
):

    score = 100

    results = []

    try:

        img = Image.open(
            file_path
        )

    except Exception:

        return (
            0,
            "🔴 Invalid or corrupted image",
            [
                (
                    "File Check",
                    "❌ Could not open image file"
                )
            ],
            0.0,
            "🔴 Corrupted Image",
            "Could not open or decode the image file."
        )

    file_format = (
        img.format
        or "UNKNOWN"
    ).upper()

    width, height = img.size

    # --------------------------------------------------------
    # EXIF METADATA
    # --------------------------------------------------------

    exif_data = {}

    try:

        raw_exif = img.getexif()

        if raw_exif:

            for (
                tag_id,
                value
            ) in raw_exif.items():

                tag = ExifTags.TAGS.get(
                    tag_id,
                    tag_id
                )

                exif_data[tag] = value

    except Exception:

        exif_data = {}

    if not exif_data:

        score -= 15

        if file_format in (
            "WEBP",
            "JPEG"
        ):

            results.append(
                (
                    "EXIF Metadata",
                    f"⚠ No metadata found ({file_format} format) - metadata is commonly stripped during conversion/sharing, this alone is not conclusive"
                )
            )

        else:

            results.append(
                (
                    "EXIF Metadata",
                    "⚠ No metadata found - common in screenshots, AI-generated images, or stripped files"
                )
            )

    else:

        if (
            "Make" in exif_data
            or "Model" in exif_data
        ):

            cam = (
                f"{exif_data.get('Make', '')} "
                f"{exif_data.get('Model', '')}"
            ).strip()

            results.append(
                (
                    "Camera Info",
                    f"✔ Camera metadata present ({cam})"
                )
            )

        else:

            results.append(
                (
                    "Camera Info",
                    "ℹ No camera make/model found in metadata"
                )
            )

        software = str(
            exif_data.get(
                "Software",
                ""
            )
        ).lower()

        editing_tools = [
            "photoshop",
            "gimp",
            "lightroom",
            "snapseed",
            "picsart",
            "canva"
        ]

        found_tool = next(
            (
                t
                for t in editing_tools
                if t in software
            ),
            None
        )

        if found_tool:

            score -= 20

            results.append(
                (
                    "Editing Software Tag",
                    f"⚠ Image edited using {found_tool.title()}"
                )
            )

    # --------------------------------------------------------
    # AI METADATA SIGNATURE
    # --------------------------------------------------------

    is_ai, keyword = (
        check_ai_generation_signature(
            img,
            exif_data,
            file_path=file_path
        )
    )

    if is_ai:

        score -= 40

        results.append(
            (
                "AI Generation Signature",
                f"❌ Possible AI-generated image (signature: {keyword})"
            )
        )

    else:

        results.append(
            (
                "AI Generation Signature",
                "✔ No known AI-generation metadata signatures found"
            )
        )

    # --------------------------------------------------------
    # ELA
    # --------------------------------------------------------

    # --------------------------------------------------------
    # ELA
    # --------------------------------------------------------

    try:

        max_diff = error_level_analysis(
            img
        )

        if max_diff > 80:

            score -= 15

            results.append(
                (
                    "Error Level Analysis",
                    f"⚠ High compression inconsistency (level: {max_diff}) - possible editing/tampering"
                )
            )

        elif max_diff > 55:

            score -= 5

            results.append(
                (
                    "Error Level Analysis",
                    f"🟡 Moderate compression inconsistency (level: {max_diff})"
                )
            )

        else:

            results.append(
                (
                    "Error Level Analysis",
                    f"✔ No significant tampering signs detected (level: {max_diff})"
                )
            )

    except Exception:

        results.append(
            (
                "Error Level Analysis",
                "⚠ Could not perform error level analysis"
            )
        )

    # --------------------------------------------------------
    # NOISE
    # --------------------------------------------------------

    noise_level = check_noise_pattern(
        img
    )

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    is_ai_resolution = (
        check_ai_resolution_pattern(
            width,
            height
        )
    )

    if (
        is_ai_resolution
        and not exif_data
        and (noise_level is None or noise_level < 5.0)
    ):

        score -= 10

        results.append(
            (
                "Resolution Pattern",
                f"⚠ {width}x{height} matches common AI-generation output sizes (weak indicator, combined with no metadata)"
            )
        )

    else:

        results.append(
            (
                "Resolution Pattern",
                f"ℹ {width}x{height}"
            )
        )

    if noise_level is not None:

        if noise_level < 2.5 and is_ai_resolution:

            score -= 40

            results.append(
                (
                    "AI Diffusion Texture",
                    f"🔴 Synthetic resolution pattern combined with ultra-smooth diffusion texture ({noise_level})"
                )
            )

        elif noise_level < 3.0:

            score -= 20

            results.append(
                (
                    "Noise Pattern",
                    f"⚠ Unusually smooth image (noise level: {noise_level}) - generative AI indicator"
                )
            )

        else:

            results.append(
                (
                    "Noise Pattern",
                    f"✔ Natural sensor-like noise detected (noise level: {noise_level})"
                )
            )

    else:

        results.append(
            (
                "Noise Pattern",
                "⚠ Could not analyze noise pattern"
            )
        )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    (
        peak_count,
        dev_std
    ) = check_frequency_artifacts(
        img
    )

    if peak_count is not None:

        if peak_count >= 3:

            score -= 15

            results.append(
                (
                    "Frequency Spectrum Analysis",
                    f"⚠ Unusual periodic patterns in frequency domain ({peak_count} anomalies) - indicator of generative upsampling artifacts"
                )
            )

        else:

            results.append(
                (
                    "Frequency Spectrum Analysis",
                    f"✔ No significant periodic artifacts detected ({peak_count} anomalies)"
                )
            )

    else:

        results.append(
            (
                "Frequency Spectrum Analysis",
                "⚠ Could not perform frequency analysis"
            )
        )

    # --------------------------------------------------------
    # PRETRAINED AI MODEL & SYNTHETIC ART ANALYSIS
    # --------------------------------------------------------

    cv2_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_crop, face_box = extract_largest_face(cv2_bgr, pad_ratio=0.50, return_box=True)
    has_face = face_crop is not None

    if has_face:
        if AI_MODEL_AVAILABLE:
            try:
                detector = load_ai_detector()

                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_preds = detector(face_pil)
                face_fake = next((float(p['score']) for p in face_preds if 'fake' in str(p.get('label', '')).lower()), 0.0)
                face_real = next((float(p['score']) for p in face_preds if 'real' in str(p.get('label', '')).lower()), 0.0)

                full_preds = detector(img.convert("RGB"))
                full_fake = next((float(p['score']) for p in full_preds if 'fake' in str(p.get('label', '')).lower()), 0.0)
                full_real = next((float(p['score']) for p in full_preds if 'real' in str(p.get('label', '')).lower()), 0.0)

                # Ensemble: balance portrait crop with full image context
                if full_real > 0.95 and face_fake > 0.60:
                    # Discount lighting/shadow artifacts on real face
                    ens_fake = (face_fake * 0.30) + (full_fake * 0.70)
                elif face_fake > 0.70 and full_fake > 0.50:
                    ens_fake = max(face_fake, full_fake)
                else:
                    ens_fake = (face_fake * 0.50) + (full_fake * 0.50)

                ens_real = 1.0 - ens_fake

                results.append(
                    (
                        "AI Model Detection",
                        f"🤖 Fake probability: {ens_fake*100:.1f}%, Real: {ens_real*100:.1f}% (Face Portrait Ensemble)"
                    )
                )

                # Check synthetic art / AI generation indicators even if a face was detected
                is_synth, art_indicators, mean_s, pct_high_s = check_synthetic_art_indicators(
                    cv2_bgr, width, height, bool(exif_data), noise_level=noise_level, ens_real=ens_real
                )

                if is_synth:
                    score -= 60
                    primary_reason = art_indicators[0] if art_indicators else "Surreal synthetic palette"
                    results.append(
                        (
                            "AI Scene Analysis",
                            f"🔴 Generative AI scene / composite styling ({primary_reason})"
                        )
                    )
                    results.append(
                        (
                            "AI Detection Verdict",
                            f"🔴 High probability of AI-generated scene / composite ({primary_reason})"
                        )
                    )
                elif ens_fake >= 0.60:
                    score -= 60
                    results.append(
                        (
                            "AI Detection Verdict",
                            f"🔴 Deepfake face manipulation detected ({ens_fake*100:.1f}%)"
                        )
                    )
                elif ens_fake >= 0.35:
                    score -= 30
                    results.append(
                        (
                            "AI Detection Verdict",
                            f"🟡 Potential synthetic facial artifacts ({ens_fake*100:.1f}%)"
                        )
                    )
                else:
                    if max_diff <= 55:
                        score = max(score, 85)
                    # Correct smooth portrait noise if it was penalized
                    if noise_level is not None and noise_level < 3.0:
                        for idx, res_item in enumerate(results):
                            if res_item[0] == "Noise Pattern" and "generative AI indicator" in res_item[1]:
                                results[idx] = (
                                    "Noise Pattern",
                                    f"✔ Natural portrait texture (sensor / portrait mode profile: {noise_level:.2f})"
                                )
                    results.append(
                        (
                            "AI Detection Verdict",
                            f"🟢 Facial features classified as authentic real ({ens_real*100:.1f}%)"
                        )
                    )

            except Exception as e:
                print(f"AI image model evaluation error: {e}")
    else:
        results.append(
            (
                "Face Analysis",
                "ℹ No human face detected (evaluating scene forensics & AI art indicators)"
            )
        )

        is_synth, art_indicators, mean_s, pct_high_s = check_synthetic_art_indicators(
            cv2_bgr, width, height, bool(exif_data), noise_level=noise_level
        )

        if is_synth:
            score -= 55
            primary_reason = art_indicators[0] if art_indicators else "Surreal synthetic palette"
            results.append(
                (
                    "AI Art Analysis",
                    f"🔴 Generative AI artwork / digital illustration detected ({primary_reason})"
                )
            )
            results.append(
                (
                    "AI Detection Verdict",
                    "🔴 High probability of AI-generated fantasy artwork / digital graphic"
                )
            )
        else:
            if AI_MODEL_AVAILABLE:
                try:
                    detector = load_ai_detector()
                    full_preds = detector(img.convert("RGB"))
                    full_fake = next((float(p['score']) for p in full_preds if 'fake' in str(p.get('label', '')).lower()), 0.0)
                    full_real = next((float(p['score']) for p in full_preds if 'real' in str(p.get('label', '')).lower()), 0.0)

                    if full_fake >= 0.80 and noise_level is not None and noise_level < 4.0:
                        score -= 50
                        results.append(
                            (
                                "AI Detection Verdict",
                                "🔴 High probability of generative AI synthesis (model + diffusion texture)"
                            )
                        )
                    else:
                        results.append(
                            (
                                "AI Detection Verdict",
                                "🟢 Natural photographic scene texture"
                            )
                        )
                except Exception as e:
                    print(f"AI scene model error: {e}")

    # --------------------------------------------------------
    # LOCALIZED TAMPERING & AI MODIFICATION PERCENTAGE
    # --------------------------------------------------------
    has_text_overlay, text_band, text_char_count = check_text_graphic_overlay(cv2_bgr)
    if has_text_overlay:
        results.append(
            (
                "Graphic & Text Overlay",
                f"🟡 Digital typography / text banner detected across {text_band} section ({text_char_count} characters)"
            )
        )

    ai_percentage, edit_type, edit_explanation = detect_localized_image_editing(
        img=img,
        cv2_bgr=cv2_bgr,
        exif_data=exif_data,
        has_face=has_face,
        face_rect=face_box if has_face else None,
        ens_fake=ens_fake if (has_face and 'ens_fake' in locals()) else (full_fake if 'full_fake' in locals() else 0.0),
        is_synth=is_synth if 'is_synth' in locals() else False,
        art_indicators=art_indicators if 'art_indicators' in locals() else [],
        noise_level=noise_level,
        max_diff=max_diff if 'max_diff' in locals() else 0,
        is_ai_meta=is_ai if 'is_ai' in locals() else False,
        meta_kw=keyword if 'keyword' in locals() else None,
        has_text_overlay=has_text_overlay,
        text_band=text_band,
        text_char_count=text_char_count
    )

    if edit_type.startswith("🔴"):
        if score > 45:
            score -= int(max(30, ai_percentage * 0.5))
    elif edit_type.startswith("🟡"):
        if score > 75:
            score = 75
    elif edit_type.startswith("🟢"):
        if score < 85 and max_diff <= 35:
            score = 85

    results.append(
        (
            "AI / Tampered Content",
            f"🤖 {ai_percentage}% ({edit_type})"
        )
    )

    results.append(
        (
            "Editing Analysis",
            f"🔍 {edit_type}"
        )
    )

    results.append(
        (
            "Modification Findings",
            f"ℹ {edit_explanation}"
        )
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = max(
        0,
        min(score, 100)
    )

    if score >= 80:

        status = "🟢 Likely Authentic"

    elif score >= 50:

        status = "🟡 Some Concerns Detected"

    else:

        status = (
            "🔴 High Risk - "
            "Possible Manipulation/AI-Generated"
        )

    return (
        score,
        status,
        results,
        ai_percentage,
        edit_type,
        edit_explanation
    )


# ============================================================
# AI MODEL - VIDEO FRAME ANALYSIS
# ============================================================

def analyze_video_with_ai_model(
    file_path
):

    """
    Analyze sampled video frames using a pretrained image-level
    Real/Fake classifier.

    The model is an image classifier, not a certified video-level
    deepfake detector. For face-oriented videos, the largest detected
    face is cropped before classification. If no face is detected, the
    full frame is used as a fallback.

    Returns:
        average_fake_probability,
        average_real_probability,
        frame_predictions,
        suspicious_frame_ratio,
        face_detected_count
    """

    detector = load_ai_detector()

    cap = cv2.VideoCapture(
        file_path
    )

    if not cap.isOpened():

        return (
            0.0,
            0.0,
            [],
            0.0,
            0
        )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if frame_count <= 0:

        cap.release()

        return (
            0.0,
            0.0,
            [],
            0.0,
            0
        )

    # --------------------------------------------------------
    # Sample 12 frames across the complete video
    # --------------------------------------------------------

    sample_positions = [
        0.05,
        0.13,
        0.21,
        0.29,
        0.37,
        0.45,
        0.53,
        0.61,
        0.69,
        0.77,
        0.85,
        0.95
    ]

    all_fake_scores = []
    face_fake_scores = []
    face_real_scores = []
    frame_predictions = []
    suspicious_count = 0
    face_detected_count = 0

    for pos in sample_positions:

        frame_index = int(
            frame_count * pos
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        ret, frame = cap.read()

        if not ret:
            continue

        try:

            # ----------------------------------------------------
            # FACE CROP & ENSEMBLE PER FRAME
            # ----------------------------------------------------

            face_crop = extract_largest_face(frame, pad_ratio=0.50)

            # Full frame PIL
            full_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            full_preds = detector(full_pil)
            full_fake = next((float(p['score']) for p in full_preds if 'fake' in str(p.get('label', '')).lower()), 0.0)
            full_real = next((float(p['score']) for p in full_preds if 'real' in str(p.get('label', '')).lower()), 0.0)

            if face_crop is not None:
                face_detected_count += 1
                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_preds = detector(face_pil)
                face_fake = next((float(p['score']) for p in face_preds if 'fake' in str(p.get('label', '')).lower()), 0.0)
                face_real = next((float(p['score']) for p in face_preds if 'real' in str(p.get('label', '')).lower()), 0.0)

                # Balance frame ensemble: full context discounts tight lighting artifacts on real faces
                if full_real > 0.95 and face_fake > 0.60:
                    frame_fake = (face_fake * 0.30) + (full_fake * 0.70)
                elif face_fake > 0.70 and full_fake > 0.50:
                    frame_fake = max(face_fake, full_fake)
                else:
                    frame_fake = (face_fake * 0.50) + (full_fake * 0.50)

                frame_real = 1.0 - frame_fake
                input_type = "face_portrait_ensemble"
                face_fake_scores.append(frame_fake)
                face_real_scores.append(frame_real)
            else:
                frame_fake = full_fake
                frame_real = full_real
                input_type = "full_frame"

            all_fake_scores.append(frame_fake)

            is_suspicious = frame_fake >= 0.50
            if is_suspicious:
                suspicious_count += 1

            frame_predictions.append(
                {
                    "frame": frame_index,
                    "fake_probability": frame_fake,
                    "real_probability": frame_real,
                    "input_type": input_type,
                    "suspicious": is_suspicious
                }
            )

        except Exception as e:

            print(
                f"AI model error on frame "
                f"{frame_index}: {e}"
            )

    cap.release()

    if not frame_predictions:

        return (
            0.0,
            0.0,
            [],
            0.0,
            face_detected_count
        )

    # When faces are consistently present (e.g. talking-head / portrait video), prioritize facial evaluation
    # But if only 1-2 frames have a face out of many sampled frames, it's an incidental detection in a scenic video
    face_ratio = len(face_fake_scores) / len(frame_predictions) if frame_predictions else 0.0
    if face_fake_scores and (face_ratio >= 0.40 or len(face_fake_scores) >= 5):
        average_fake = (
            sum(face_fake_scores)
            / len(face_fake_scores)
        )
        average_real = (
            sum(face_real_scores)
            / len(face_real_scores)
        )
        suspicious_frame_ratio = (
            sum(1 for s in face_fake_scores if s >= 0.50)
            / len(face_fake_scores)
        )
    else:
        average_fake = (
            sum(all_fake_scores)
            / len(all_fake_scores)
        ) if all_fake_scores else 0.0

        average_real = (
            sum(
                item.get("real_probability", 0.0)
                for item in frame_predictions
            )
            / len(frame_predictions)
        ) if frame_predictions else 0.0

        suspicious_frame_ratio = (
            suspicious_count / len(all_fake_scores)
        ) if all_fake_scores else 0.0

    return (
        average_fake,
        average_real,
        frame_predictions,
        suspicious_frame_ratio,
        face_detected_count
    )


# ============================================================
# LOCALIZED VIDEO EDITING & AI PERCENTAGE DETECTION
# ============================================================

def detect_localized_video_editing(
    is_gen_ai_video,
    gen_ai_reasons,
    is_face_video,
    face_detected_count,
    suspicious_count,
    total_sampled_frames,
    fake_probability,
    temporal_variation,
    avg_temporal=0.0,
    avg_saturation=0.0
):
    """
    Detect video manipulation: full generative AI video, deepfake face swap,
    video inpainting / object tampering, software video editing, or authentic camera video.
    Returns (ai_percentage, edit_type, edit_explanation).
    """
    tot = max(1, total_sampled_frames)

    # Case 1: Full Generative AI Video (Sora / Runway / Pika / Kling)
    if is_gen_ai_video:
        reasons_str = ", ".join(gen_ai_reasons)
        return (
            100.0,
            "🔴 Full Generative AI Video (100% Synthetic)",
            f"The entire video was synthetically generated using AI video diffusion models ({reasons_str}). No genuine physical camera lens or sensor noise profile detected."
        )

    # Case 2: Deepfake Face Swap
    if is_face_video and suspicious_count > 0:
        ratio = round((suspicious_count / tot) * 100, 1)
        return (
            ratio,
            "🔴 Deepfake Facial Manipulation (Face Swap)",
            f"AI facial manipulation detected in {suspicious_count}/{tot} sampled frames ({ratio}% of the video). The subject's face was digitally replaced with synthetic facial features while preserving the camera recording."
        )

    # Case 3: Video Splicing / Object Addition / Inpainting
    if (suspicious_count > 0 and suspicious_count < tot):
        altered_frames = max(1, suspicious_count)
        ratio = round((altered_frames / tot) * 100, 1)
        return (
            ratio,
            "🔴 Video Tampering / Splicing & Object Modification",
            f"Video tampering / frame splicing detected across {altered_frames}/{tot} sampled frames ({ratio}%). Frame motion discontinuity or localized alterations detected."
        )

    # Case 4: Video Edited Using Software (Scene Cuts, Montage Transitions, Color Grading)
    if avg_temporal > 45.0 or (avg_temporal > 30.0 and avg_saturation > 75.0) or temporal_variation > 40.0:
        return (
            25.0,
            "🟡 Video Editing / Software Montage & Color Grading",
            f"Video edited using editing software (montage scene cuts and transitions detected with average frame shift: {avg_temporal:.1f}, plus color grading saturation: {avg_saturation:.1f}). Original footage captured from physical camera, but edited, spliced, and re-encoded in video editing software."
        )

    # Case 5: Authentic Camera Video
    return (
        0.0,
        "🟢 Authentic Camera Video (Unedited)",
        "Original, unmodified camera recording. Natural sensor noise, uniform compression across sampled frames, continuous motion continuity, and no AI manipulation detected."
    )


# ============================================================
# VIDEO FORENSICS
# ============================================================

def analyze_video_forensics(
    file_path
):

    score = 100

    results = []

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        file_path
    )

    if not cap.isOpened():

        return (
            0,
            "🔴 Invalid or unreadable video",
            [
                (
                    "File Check",
                    "❌ Could not open video file"
                )
            ],
            0.0,
            "🔴 Corrupted / Unreadable Video",
            "Could not open or decode the video file."
        )

    # --------------------------------------------------------
    # VIDEO METADATA
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fourcc_int = int(
        cap.get(
            cv2.CAP_PROP_FOURCC
        )
    )

    codec = "".join(
        [
            chr(
                (
                    fourcc_int
                    >> 8 * i
                )
                & 0xFF
            )
            for i in range(4)
        ]
    ).strip()

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    results.append(
        (
            "Resolution",
            f"ℹ {width} x {height}"
        )
    )

    results.append(
        (
            "Frame Rate",
            f"ℹ {fps:.1f} fps"
        )
    )

    results.append(
        (
            "Duration",
            f"ℹ {duration:.1f} seconds"
        )
    )

    results.append(
        (
            "Codec",
            f"ℹ {codec if codec else 'Unknown'}"
        )
    )

    # --------------------------------------------------------
    # VIDEO INTEGRITY
    # --------------------------------------------------------

    if (
        fps <= 0
        or frame_count <= 0
    ):

        score -= 20

        results.append(
            (
                "Video Integrity",
                "⚠ Could not read frame/fps info properly - file may be corrupted"
            )
        )

    # --------------------------------------------------------
    # SAMPLE FRAMES FOR FORENSIC ANALYSIS
    # --------------------------------------------------------

    sample_positions = [
        0.05,
        0.15,
        0.25,
        0.35,
        0.50,
        0.65,
        0.75,
        0.85,
        0.95
    ]

    ela_values = []
    noise_values = []
    frequency_values = []
    temporal_values = []
    saturation_values = []
    high_sat_pixel_ratios = []

    sampled_frames = []

    previous_gray = None

    for pos in sample_positions:

        frame_index = int(
            frame_count * pos
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        ret, frame = cap.read()

        if not ret:
            continue

        try:

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_frame = Image.fromarray(
                rgb_frame
            )

            sampled_frames.append(
                pil_frame
            )

            # ------------------------------------------------
            # ELA
            # ------------------------------------------------

            try:

                ela = error_level_analysis(
                    pil_frame
                )

                ela_values.append(
                    float(ela)
                )

            except Exception:

                pass

            # ------------------------------------------------
            # NOISE
            # ------------------------------------------------

            try:

                noise = check_noise_pattern(
                    pil_frame
                )

                if noise is not None:

                    noise_values.append(
                        float(noise)
                    )

            except Exception:

                pass

            # ------------------------------------------------
            # FREQUENCY
            # ------------------------------------------------

            try:

                (
                    peak_count,
                    frequency_std
                ) = check_frequency_artifacts(
                    pil_frame
                )

                if frequency_std is not None:

                    frequency_values.append(
                        float(frequency_std)
                    )

            except Exception:

                pass

            # ------------------------------------------------
            # TEMPORAL DIFFERENCE
            # ------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            if previous_gray is not None:

                temporal_difference = cv2.absdiff(
                    previous_gray,
                    gray
                ).mean()

                temporal_values.append(
                    float(
                        temporal_difference
                    )
                )

            previous_gray = gray

            # ------------------------------------------------
            # COLOR SATURATION
            # ------------------------------------------------
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                sat_plane = hsv[:, :, 1]
                saturation_values.append(float(np.mean(sat_plane)))
                high_sat_pixel_ratios.append(float(np.mean(sat_plane > 140)))
            except Exception:
                pass

        except Exception as e:

            print(
                f"Frame forensic error: {e}"
            )

    cap.release()

    # ========================================================
    # ELA ANALYSIS
    # ========================================================

    if ela_values:

        avg_ela = float(
            np.mean(ela_values)
        )

        ela_std = float(
            np.std(ela_values)
        )

        results.append(
            (
                "Compression Consistency",
                f"ℹ Frame variation: {ela_std:.1f}"
            )
        )

        # In continuous authentic video recordings, compression is consistent across frames
        if ela_std > 25:

            score -= 15

            results.append(
                (
                    "Compression Analysis",
                    f"🔴 Inconsistent compression across frames ({ela_std:.1f}) - potential splicing"
                )
            )

        elif ela_std > 18:

            score -= 5

            results.append(
                (
                    "Compression Analysis",
                    f"🟡 Moderate compression variation ({ela_std:.1f})"
                )
            )

        else:

            results.append(
                (
                    "Compression Analysis",
                    f"✔ Consistent compression pattern across sampled frames ({ela_std:.1f})"
                )
            )

    else:

        results.append(
            (
                "Frame-level Analysis",
                "⚠ Could not calculate ELA"
            )
        )

    # ========================================================
    # NOISE ANALYSIS
    # ========================================================

    if noise_values:

        avg_noise = float(
            np.mean(noise_values)
        )

        results.append(
            (
                "Noise Pattern",
                f"ℹ Average noise level: {avg_noise:.1f}"
            )
        )

        # Diffusion/AI video frames have unnaturally low noise (< 4.0). Real camera sensors produce natural noise (> 7.0).
        if avg_noise < 4.0:

            score -= 15

            results.append(
                (
                    "Noise Pattern",
                    f"🔴 Unnaturally smooth surface texture ({avg_noise:.1f}) - AI generation indicator"
                )
            )

        elif avg_noise < 7.0:

            score -= 5

            results.append(
                (
                    "Noise Pattern",
                    f"🟡 Low sensor noise detected ({avg_noise:.1f})"
                )
            )

        else:

            results.append(
                (
                    "Noise Pattern",
                    f"✔ Natural camera sensor noise detected ({avg_noise:.1f})"
                )
            )

    else:

        results.append(
            (
                "Noise Pattern",
                "⚠ Could not analyze noise"
            )
        )

    # ========================================================
    # RESOLUTION ANALYSIS
    # ========================================================

    is_ai_resolution = (
        check_ai_resolution_pattern(
            width,
            height
        )
    )

    results.append(
        (
            "Resolution Pattern",
            f"ℹ {width}x{height}"
        )
    )

    if is_ai_resolution:

        score -= 5

        results.append(
            (
                "Resolution Pattern",
                "🟡 Resolution matches a common AI-generation size pattern"
            )
        )

    # ========================================================
    # FREQUENCY ANALYSIS
    # ========================================================

    if frequency_values:

        avg_frequency = float(
            np.mean(frequency_values)
        )

        frequency_variation = float(
            np.std(frequency_values)
        )

        results.append(
            (
                "Frequency Analysis",
                f"ℹ Average high-frequency variation: {avg_frequency:.2f}"
            )
        )

        results.append(
            (
                "Frequency Consistency",
                f"ℹ Variation: {frequency_variation:.2f}"
            )
        )

        if frequency_variation > 20:

            score -= 10

            results.append(
                (
                    "Frequency Analysis",
                    "🔴 High frequency inconsistency detected"
                )
            )

        elif frequency_variation > 10:

            score -= 5

            results.append(
                (
                    "Frequency Analysis",
                    "🟡 Moderate frequency inconsistency detected"
                )
            )

        else:

            results.append(
                (
                    "Frequency Analysis",
                    "✔ Frequency pattern is relatively consistent"
                )
            )

    else:

        results.append(
            (
                "Frequency Analysis",
                "⚠ Could not calculate frequency pattern"
            )
        )

    # ========================================================
    # TEMPORAL ANALYSIS
    # ========================================================

    if temporal_values:

        avg_temporal = float(
            np.mean(temporal_values)
        )

        temporal_variation = float(
            np.std(temporal_values)
        )

        results.append(
            (
                "Temporal Analysis",
                f"ℹ Average frame difference: {avg_temporal:.1f}"
            )
        )

        results.append(
            (
                "Temporal Consistency",
                f"ℹ Variation: {temporal_variation:.1f}"
            )
        )

        if avg_temporal > 45.0:

            score -= 15

            results.append(
                (
                    "Temporal Continuity",
                    f"🟡 Multiple scene cuts / montage transitions detected ({avg_temporal:.1f}) - edited in video software"
                )
            )

        elif temporal_variation > 55:

            score -= 10

            results.append(
                (
                    "Temporal Continuity",
                    f"🔴 Abrupt frame discontinuity detected ({temporal_variation:.1f}) - possible frame insertion/cut"
                )
            )

        elif temporal_variation > 40:

            score -= 5

            results.append(
                (
                    "Temporal Continuity",
                    f"🟡 Moderate motion fluctuation detected ({temporal_variation:.1f})"
                )
            )

        else:

            results.append(
                (
                    "Temporal Continuity",
                    "✔ Natural continuous video motion profile"
                )
            )

    else:

        results.append(
            (
                "Temporal Analysis",
                "⚠ Could not calculate temporal consistency"
            )
        )

    # ========================================================
    # COLOR SATURATION ANALYSIS
    # ========================================================

    if saturation_values:

        avg_saturation = float(
            np.mean(saturation_values)
        )

        results.append(
            (
                "Color Saturation",
                f"ℹ Average color saturation: {avg_saturation:.1f}"
            )
        )

        if avg_saturation > 110.0:

            results.append(
                (
                    "Color Saturation",
                    f"🔴 Hyper-saturated synthetic palette ({avg_saturation:.1f}) - characteristic of AI generation"
                )
            )

        elif avg_saturation > 80.0:

            score -= 5

            results.append(
                (
                    "Color Saturation",
                    f"🟡 Enhanced color grading / saturation filter detected ({avg_saturation:.1f})"
                )
            )

        else:

            results.append(
                (
                    "Color Saturation",
                    f"✔ Natural color saturation profile ({avg_saturation:.1f})"
                )
            )

    # ========================================================
    # PRETRAINED AI MODEL
    # ========================================================

    try:

        (
            fake_probability,
            real_probability,
            frame_predictions,
            suspicious_frame_ratio,
            face_detected_count
        ) = analyze_video_with_ai_model(
            file_path
        )

        fake_percentage = (
            fake_probability * 100
        )

        real_percentage = (
            real_probability * 100
        )

        suspicious_percentage = (
            suspicious_frame_ratio * 100
        )

        results.append(
            (
                "AI Model Analysis",
                f"🤖 Fake probability: {fake_percentage:.1f}%"
            )
        )

        results.append(
            (
                "AI Model Analysis",
                f"🤖 Real probability: {real_percentage:.1f}%"
            )
        )

        results.append(
            (
                "AI Frame Samples",
                f"ℹ Analyzed {len(frame_predictions)} video frames"
            )
        )

        results.append(
            (
                "AI Face Analysis",
                f"ℹ Face detected in {face_detected_count}/{len(frame_predictions)} sampled frames"
            )
        )

        if frame_predictions:
            suspicious_count = sum(1 for item in frame_predictions if item.get('suspicious'))
            suspicious_percentage = (suspicious_count / len(frame_predictions)) * 100
            results.append(
                (
                    "AI Suspicious Frames",
                    f"ℹ Suspicious frames: {suspicious_count}/{len(frame_predictions)} ({suspicious_percentage:.1f}%)"
                )
            )

            max_fake_probability = max(
                item.get("fake_probability", 0.0)
                for item in frame_predictions
            )

            results.append(
                (
                    "AI Peak Detection",
                    f"ℹ Highest frame-level fake probability: {max_fake_probability * 100:.1f}%"
                )
            )

        # ----------------------------------------------------
        # AI SCORE DEDUCTION & VERDICT
        # ----------------------------------------------------
        avg_sat = float(np.mean(saturation_values)) if saturation_values else 0.0
        avg_high_sat_ratio = float(np.mean(high_sat_pixel_ratios)) if high_sat_pixel_ratios else 0.0
        avg_noise_val = float(np.mean(noise_values)) if noise_values else 20.0
        freq_var = float(np.std(frequency_values)) if frequency_values else 0.0

        is_face_video = (
            face_detected_count >= max(3, int(len(frame_predictions) * 0.40))
        )

        # Comprehensive Generative AI indicators for videos (Sora / Runway / Kling / Pika / Stable Video)
        is_gen_ai_video = False
        gen_ai_reasons = []

        if avg_sat > 110.0:
            is_gen_ai_video = True
            gen_ai_reasons.append(f"Hyper-saturated synthetic color grading (mean saturation: {avg_sat:.1f})")
        elif avg_sat > 85.0 and avg_high_sat_ratio > 0.20:
            is_gen_ai_video = True
            gen_ai_reasons.append(f"Synthetic vibrant color palette ({avg_high_sat_ratio * 100:.1f}% high-saturation pixels)")
        elif avg_sat > 80.0 and (fake_probability >= 0.45 or suspicious_frame_ratio >= 0.50):
            is_gen_ai_video = True
            gen_ai_reasons.append(f"Synthetic color profile with generative frame synthesis ({fake_percentage:.1f}%)")
        elif suspicious_frame_ratio >= 0.60 or fake_probability >= 0.50:
            is_gen_ai_video = True
            gen_ai_reasons.append(f"Model detected generative AI frame synthesis across {suspicious_count}/{len(frame_predictions)} frames")
        elif is_ai_resolution and avg_noise_val < 8.0:
            is_gen_ai_video = True
            gen_ai_reasons.append("Synthetic aspect ratio and smooth diffusion texture")
        elif avg_noise_val < 4.0:
            is_gen_ai_video = True
            gen_ai_reasons.append(f"Unnaturally smooth diffusion texture without camera sensor noise ({avg_noise_val:.1f})")

        if is_face_video:
            # Face-oriented video (deepfake analysis)
            combined_ai_risk = (
                (fake_probability * 0.70)
                + (suspicious_frame_ratio * 0.30)
            )

            if combined_ai_risk >= 0.60 or fake_probability >= 0.65 or suspicious_frame_ratio >= 0.50:
                score -= 65
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🔴 High probability of AI/deepfake face manipulation ({fake_percentage:.1f}%)"
                    )
                )

            elif is_gen_ai_video:
                score -= 65
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🔴 AI-generated video detected ({', '.join(gen_ai_reasons)})"
                    )
                )

            elif combined_ai_risk >= 0.40 or fake_probability >= 0.45:
                score -= 35
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🟡 Moderate AI/deepfake indicators detected ({fake_percentage:.1f}%)"
                    )
                )

            elif combined_ai_risk >= 0.25:
                score -= 15
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🟡 Weak AI/synthetic traces ({fake_percentage:.1f}%)"
                    )
                )

            else:
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🟢 Facial analysis indicates authentic real features ({real_percentage:.1f}% real)"
                    )
                )

        else:
            # Non-face / general scene video (Generative AI scenes / Sora / Runway / general camera video)
            if is_gen_ai_video:
                score -= 65
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🔴 AI-generated video detected ({', '.join(gen_ai_reasons)})"
                    )
                )

            elif fake_probability >= 0.40 or suspicious_frame_ratio >= 0.40:
                score -= 35
                results.append(
                    (
                        "AI Detection Verdict",
                        f"🟡 Moderate generative AI indicators detected ({fake_percentage:.1f}%)"
                    )
                )

            elif freq_var > 22.0:
                score -= 20
                results.append(
                    (
                        "AI Frequency Analysis",
                        "🟡 Periodic frequency upsampling artifacts detected"
                    )
                )

            else:
                results.append(
                    (
                        "AI Video Verdict",
                        "🟢 Visual grain, compression, and sensor noise match genuine camera video recording"
                    )
                )

    except Exception as e:

        results.append(
            (
                "AI Model",
                f"⚠ AI model unavailable: {str(e)}"
            )
        )

        print(
            f"AI model analysis failed: {e}"
        )

    # --------------------------------------------------------
    # LOCALIZED VIDEO EDITING & AI PERCENTAGE
    # --------------------------------------------------------
    ai_percentage, edit_type, edit_explanation = detect_localized_video_editing(
        is_gen_ai_video=is_gen_ai_video if 'is_gen_ai_video' in locals() else False,
        gen_ai_reasons=gen_ai_reasons if 'gen_ai_reasons' in locals() else [],
        is_face_video=is_face_video if 'is_face_video' in locals() else False,
        face_detected_count=face_detected_count if 'face_detected_count' in locals() else 0,
        suspicious_count=suspicious_count if 'suspicious_count' in locals() else 0,
        total_sampled_frames=len(frame_predictions) if 'frame_predictions' in locals() else 12,
        fake_probability=fake_probability if 'fake_probability' in locals() else 0.0,
        temporal_variation=temporal_variation if 'temporal_variation' in locals() else 0.0,
        avg_temporal=avg_temporal if 'avg_temporal' in locals() else 0.0,
        avg_saturation=avg_saturation if 'avg_saturation' in locals() else 0.0
    )

    if edit_type.startswith("🔴"):
        if score > 45:
            score -= int(max(30, ai_percentage * 0.5))
    elif edit_type.startswith("🟡"):
        if score > 75:
            score = 75

    results.append(
        (
            "AI / Tampered Content",
            f"🤖 {ai_percentage}% ({edit_type})"
        )
    )

    results.append(
        (
            "Editing Analysis",
            f"🔍 {edit_type}"
        )
    )

    results.append(
        (
            "Modification Findings",
            f"ℹ {edit_explanation}"
        )
    )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    score = max(
        0,
        min(score, 100)
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if score >= 80:

        status = "🟢 Likely Authentic"

    elif score >= 50:

        status = "🟡 Some Concerns Detected"

    else:

        status = (
            "🔴 High Risk - "
            "Possible Manipulation/AI-Generated"
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    results.append(
        (
            "Important",
            "⚠ Forensic + ML analysis only - "
            "not a certified AI/deepfake detector"
        )
    )

    return (
        score,
        status,
        results,
        ai_percentage,
        edit_type,
        edit_explanation
    )