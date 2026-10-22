import re
import dns.resolver

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import datetime
from urllib.parse import urlparse

from rule_engine import analyze_website, analyze_qr_content, analyze_video_forensics, analyze_image_forensics
import os
from werkzeug.utils import secure_filename
from requests.adapters import HTTPAdapter

import requests
import ssl
import socket
from bs4 import BeautifulSoup
try:
    import mysql.connector
except ImportError:
    pass

from werkzeug.security import generate_password_hash, check_password_hash
from database import get_connection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

app.secret_key = "trustlens_secret_key"


UPLOAD_FOLDER = os.path.join('/tmp', 'temp_uploads') if os.environ.get('VERCEL') else 'temp_uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}


# =========================================================
# LEGACY SSL ADAPTER
# =========================================================

class LegacySSLAdapter(HTTPAdapter):

    def init_poolmanager(self, *args, **kwargs):

        context = ssl.create_default_context()

        context.options |= 0x4

        context.check_hostname = False

        context.verify_mode = ssl.CERT_NONE

        kwargs['ssl_context'] = context

        return super().init_poolmanager(*args, **kwargs)


# =========================================================
# NORMALIZE URL
# =========================================================

def normalize_url(raw_url):

    raw_url = raw_url.strip()

    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url

    try:

        requests.get(
            "https://" + raw_url,
            timeout=5,
            headers=BROWSER_HEADERS
        )

        return "https://" + raw_url

    except Exception:
        pass

    try:

        requests.get(
            "http://" + raw_url,
            timeout=5,
            headers=BROWSER_HEADERS
        )

        return "http://" + raw_url

    except Exception:
        pass

    return "https://" + raw_url


# =========================================================
# GET DOMAIN AGE
# =========================================================

def get_domain_age(url):

    try:

        domain = urlparse(url).netloc

        domain = domain.replace("www.", "")

        rdap_url = f"https://rdap.org/domain/{domain}"

        response = requests.get(
            rdap_url,
            timeout=10
        )

        if response.status_code != 200:

            return (
                "Not Available",
                "⚠ Domain information unavailable"
            )

        data = response.json()

        creation_date = None

        for event in data.get("events", []):

            if event.get("eventAction") == "registration":

                creation_date = event.get("eventDate")

                break

        if not creation_date:

            return (
                "Not Available",
                "⚠ Registration date unavailable"
            )

        created = datetime.fromisoformat(
            creation_date.replace("Z", "+00:00")
        )

        today = datetime.now(created.tzinfo)

        age_days = (today - created).days

        age_years = age_days // 365

        if age_years >= 3:

            status = "✔ Established Domain"

        elif age_years >= 1:

            status = "🟡 Relatively New Domain"

        else:

            status = "⚠ Very New Domain"

        return f"{age_years} Years", status

    except Exception:

        return (
            "Not Available",
            "⚠ Domain age check failed"
        )


# =========================================================
# Image Forensics is imported from rule_engine

# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# SIGN UP
# =========================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        userid = request.form["userid"]

        fullname = request.form["fullname"]

        email = request.form["email"]

        password = request.form["password"]

        confirm_password = request.form["confirm_password"]


        is_ajax = request.headers.get(
            "X-Requested-With"
        ) == "XMLHttpRequest"


        if password != confirm_password:

            if is_ajax:

                return jsonify(
                    success=False,
                    message="Passwords do not match"
                ), 400

            return "Passwords do not match"


        conn = get_connection()

        cursor = conn.cursor()


        cursor.execute(
            "SELECT * FROM users WHERE email=%s OR userid=%s",
            (email, userid)
        )


        existing = cursor.fetchone()


        if existing:

            cursor.close()

            conn.close()


            if is_ajax:

                return jsonify(
                    success=False,
                    message="This account already exists. Please Sign In."
                ), 409

            return "This account already exists. Please Sign In."


        hashed_password = generate_password_hash(
            password
        )


        cursor.execute(
            """
            INSERT INTO users
            (userid, fullname, email, password)
            VALUES (%s,%s,%s,%s)
            """,
            (
                userid,
                fullname,
                email,
                hashed_password
            )
        )


        conn.commit()

        cursor.close()

        conn.close()


        if is_ajax:

            return jsonify(
                success=True,
                redirect=url_for("signin")
            )


        return redirect("/signin")


    return render_template(
        "signup.html"
    )


# =========================================================
# SIGN IN
# =========================================================

@app.route(
    "/signin",
    methods=["GET", "POST"]
)
def signin():

    if request.method == "POST":

        userid = request.form["userid"]

        password = request.form["password"]


        is_ajax = request.headers.get(
            "X-Requested-With"
        ) == "XMLHttpRequest"


        conn = get_connection()

        cursor = conn.cursor(
            dictionary=True
        )


        cursor.execute(
            "SELECT * FROM users WHERE userid=%s",
            (userid,)
        )


        user = cursor.fetchone()


        cursor.close()

        conn.close()


        if user and check_password_hash(
            user["password"],
            password
        ):

            session["userid"] = user["userid"]

            session["fullname"] = user["fullname"]


            if is_ajax:

                return jsonify(
                    success=True,
                    redirect=url_for("dashboard")
                )


            return redirect("/dashboard")


        if is_ajax:

            return jsonify(
                success=False,
                message="Invalid User ID or Password"
            ), 401


        return "Invalid User ID or Password"


    return render_template(
        "signin.html"
    )


# =========================================================
# FORGOT PASSWORD
# =========================================================

@app.route(
    '/forgot-password',
    methods=['GET', 'POST']
)
def forgot_password():

    if request.method == 'POST':

        userid = request.form.get(
            'userid'
        )

        new_password = request.form.get(
            'new_password'
        )

        confirm_password = request.form.get(
            'confirm_password'
        )


        if new_password != confirm_password:

            return render_template(
                'forgot_password.html',
                error="Passwords do not match"
            )


        connection = get_connection()

        cursor = connection.cursor()


        cursor.execute(
            "SELECT userid FROM users WHERE userid = %s",
            (userid,)
        )


        user = cursor.fetchone()


        if user is None:

            cursor.close()

            connection.close()


            return render_template(
                'forgot_password.html',
                error="User ID not found"
            )


        hashed_password = generate_password_hash(
            new_password
        )


        cursor.execute(
            "UPDATE users SET password = %s WHERE userid = %s",
            (
                hashed_password,
                userid
            )
        )


        connection.commit()

        cursor.close()

        connection.close()


        return render_template(
            'forgot_password.html',
            success="Password reset successful. Please sign in."
        )


    return render_template(
        'forgot_password.html'
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "userid" not in session:

        return redirect("/signin")


    return render_template(
        "dashboard.html",
        fullname=session["fullname"]
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/signin")


# =========================================================
# WEBSITE ANALYSIS
# =========================================================

@app.route(
    "/website",
    methods=["GET", "POST"]
)
def website():

    if request.method == "POST":

        raw_url = request.form["url"].strip()

        url = normalize_url(
            raw_url
        )


        score, status, results = analyze_website(
            url
        )


        domain_age, domain_status = get_domain_age(
            url
        )


        content_score = 0

        content_results = []

        page_title = "Not Available"


        try:

            try:

                response = requests.get(
                    url,
                    timeout=10,
                    headers=BROWSER_HEADERS
                )

                print(
                    "DEBUG: Normal request succeeded"
                )


            except Exception as e1:

                print(
                    "DEBUG: Normal request failed:",
                    type(e1).__name__,
                    str(e1)
                )


                try:

                    response = requests.get(
                        url,
                        timeout=10,
                        headers=BROWSER_HEADERS,
                        verify=False
                    )

                    print(
                        "DEBUG: verify=False succeeded"
                    )


                except Exception as e2:

                    print(
                        "DEBUG: verify=False failed:",
                        type(e2).__name__,
                        str(e2)
                    )


                    legacy_session = requests.Session()


                    legacy_session.mount(
                        "https://",
                        LegacySSLAdapter()
                    )


                    response = legacy_session.get(
                        url,
                        timeout=10,
                        headers=BROWSER_HEADERS,
                        verify=False
                    )


                    print(
                        "DEBUG: Legacy SSL succeeded"
                    )


            if response.status_code == 200:

                content_score += 20


                content_results.append(
                    (
                        "Website Reachability",
                        "✔ Website is reachable"
                    )
                )


                soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )


                if soup.title and soup.title.string:

                    page_title = soup.title.string.strip()

                    content_score += 20


                    content_results.append(
                        (
                            "Page Title",
                            "✔ Available"
                        )
                    )


                else:

                    content_results.append(
                        (
                            "Page Title",
                            "❌ Not Available"
                        )
                    )


                for element in soup(
                    [
                        "script",
                        "style",
                        "noscript"
                    ]
                ):

                    element.decompose()


                text = soup.get_text(
                    separator=" ",
                    strip=True
                )


                text_lower = text.lower()


                suspicious_keywords = [

                    "verify your account",

                    "confirm your password",

                    "enter your password",

                    "urgent action required",

                    "your account will be suspended",

                    "click here immediately",

                    "you have won",

                    "claim your prize"
                ]


                found_keywords = [

                    k
                    for k in suspicious_keywords
                    if k in text_lower
                ]


                if found_keywords:

                    content_score -= 20


                    content_results.append(
                        (
                            "Suspicious Content",
                            "⚠ Suspicious phrases detected"
                        )
                    )


                else:

                    content_score += 20


                    content_results.append(
                        (
                            "Suspicious Content",
                            "✔ No suspicious phrases detected"
                        )
                    )


                if len(text) > 100:

                    content_score += 20


                    content_results.append(
                        (
                            "Page Content",
                            "✔ Sufficient content found"
                        )
                    )


                else:

                    content_results.append(
                        (
                            "Page Content",
                            "⚠ Very little content found"
                        )
                    )


            else:

                print(
                    "DEBUG: Non-200 status code:",
                    response.status_code
                )


                content_results.append(
                    (
                        "Website Reachability",
                        "❌ Unable to access website"
                    )
                )


        except requests.exceptions.RequestException as e:

            print(
                "DEBUG: Final exception:",
                type(e).__name__,
                str(e)
            )


            content_results.append(
                (
                    "Website Reachability",
                    "❌ Connection failed"
                )
            )


        if content_score < 0:

            content_score = 0


        if content_score > 100:

            content_score = 100


        if domain_age != "Not Available":

            try:

                age_years = int(
                    domain_age.split()[0]
                )


                if age_years >= 5:

                    domain_age_score = 100


                elif age_years >= 3:

                    domain_age_score = 80


                elif age_years >= 1:

                    domain_age_score = 60


                else:

                    domain_age_score = 30


            except:

                domain_age_score = 0


        else:

            domain_age_score = 0


        overall_score = int(
            (score * 0.5)
            + (content_score * 0.3)
            + (domain_age_score * 0.2)
        )


        if overall_score >= 80:

            overall_status = "🟢 Safe Website"


        elif overall_score >= 50:

            overall_status = "🟡 Medium Risk"


        else:

            overall_status = "🔴 Suspicious Website"


        return render_template(

            "website.html",

            url=url,

            score=score,

            status=status,

            results=results,

            content_score=content_score,

            content_results=content_results,

            page_title=page_title,

            domain_age=domain_age,

            domain_status=domain_status,

            overall_score=overall_score,

            overall_status=overall_status
        )


    return render_template(
        "website.html"
    )


# =========================================================
# PASSWORD ANALYSIS
# =========================================================

@app.route(
    "/password",
    methods=["GET", "POST"]
)
def password():

    if request.method == "POST":

        password = request.form.get(
            "password",
            ""
        ).strip()


        if not password:

            return render_template(
                "password.html",
                error="Please enter a password."
            )


        score = 0

        results = []


        common_passwords = {
            "password", "password123", "password1234", "password12345",
            "password1!", "admin", "admin123", "admin123!", "12345678",
            "123456789", "1234567890", "qwerty123", "qwertyuiop",
            "welcome123", "welcome@123", "letmein123", "iloveyou123"
        }

        is_common = password.lower() in common_passwords

        if len(password) >= 12:

            score += 20

            results.append(
                (
                    "Password Length",
                    f"✔ Excellent length ({len(password)} characters)"
                )
            )

        elif len(password) >= 8:

            score += 20

            results.append(
                (
                    "Password Length",
                    f"✔ Acceptable length ({len(password)} characters)"
                )
            )

        else:

            results.append(
                (
                    "Password Length",
                    f"❌ Too short ({len(password)} characters - minimum 8 required)"
                )
            )


        if any(
            char.isupper()
            for char in password
        ):

            score += 20

            results.append(
                (
                    "Uppercase Letter",
                    "✔ Present"
                )
            )

        else:

            results.append(
                (
                    "Uppercase Letter",
                    "❌ Not Present"
                )
            )


        if any(
            char.islower()
            for char in password
        ):

            score += 20

            results.append(
                (
                    "Lowercase Letter",
                    "✔ Present"
                )
            )

        else:

            results.append(
                (
                    "Lowercase Letter",
                    "❌ Not Present"
                )
            )


        if any(
            char.isdigit()
            for char in password
        ):

            score += 20

            results.append(
                (
                    "Number",
                    "✔ Present"
                )
            )

        else:

            results.append(
                (
                    "Number",
                    "❌ Not Present"
                )
            )


        if any(
            not char.isalnum()
            for char in password
        ):

            score += 20

            results.append(
                (
                    "Special Character",
                    "✔ Present"
                )
            )

        else:

            results.append(
                (
                    "Special Character",
                    "❌ Not Present"
                )
            )


        if is_common:
            score = 15
            results.append(
                (
                    "Dictionary Check",
                    "❌ Widely used / easily guessable password pattern detected"
                )
            )
        else:
            results.append(
                (
                    "Dictionary Check",
                    "✔ Not a common easily guessable dictionary pattern"
                )
            )

        # A password shorter than 8 chars can never be strong
        if len(password) < 8:
            score = min(score, 45)

        score = max(0, min(score, 100))

        if score >= 80:

            status = "🟢 Strong Password"

            recommendation = (
                "Your password has good security characteristics."
            )


        elif score >= 50:

            status = "🟡 Medium Strength Password"

            recommendation = (
                "Consider adding more character types and using a longer password."
            )


        else:

            status = "🔴 Weak Password"

            recommendation = (
                "Use a longer password (at least 8-12 characters) with uppercase, lowercase, numbers, and symbols. Avoid common dictionary words."
            )


        if score == 100:

            crack_time = "Extremely Difficult to Guess"

            resistance = "🟢 Very Strong"


        elif score >= 80:

            crack_time = "Very Difficult to Guess"

            resistance = "🟢 Strong"


        elif score >= 50:

            crack_time = "Moderately Difficult to Guess"

            resistance = "🟡 Medium"


        else:

            crack_time = "Relatively Easy to Guess"

            resistance = "🔴 Weak"


        return render_template(

            "password.html",

            score=score,

            status=status,

            results=results,

            recommendation=recommendation,

            crack_time=crack_time,

            resistance=resistance
        )


    return render_template(
        "password.html"
    )


# =========================================================
# EMAIL ANALYSIS
# =========================================================

@app.route(
    '/email',
    methods=['GET', 'POST']
)
def email_analysis():

    if request.method == 'POST':

        email = request.form.get(
            'email',
            ''
        ).strip().lower()


        score = 0

        reasons = []


        domain_age = None

        domain_status = None


        # =====================================================
        # 1. EMAIL FORMAT - 15 POINTS
        # =====================================================

        email_pattern = (
            r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
        )


        if re.match(
            email_pattern,
            email
        ):

            score += 15

            reasons.append(
                "✔ Email format is valid"
            )


        else:

            reasons.append(
                "❌ Invalid email format"
            )


            return render_template(

                'email.html',

                email=email,

                score=0,

                status="🔴 Suspicious Email",

                reasons=reasons,

                domain_age=None,

                domain_status=None
            )


        # =====================================================
        # SPLIT EMAIL
        # =====================================================

        username, domain = email.split(
            '@',
            1
        )


        username = username.strip()

        domain = domain.strip()


        # =====================================================
        # 2. USERNAME ANALYSIS - 15 POINTS
        # =====================================================

        if len(username) < 4:

            score += 5

            reasons.append(
                "⚠ Email username is very short"
            )


        elif len(username) < 6:

            score += 10

            reasons.append(
                "🟡 Email username is relatively short"
            )


        else:

            score += 15

            reasons.append(
                "✔ Email username length looks normal"
            )


        digit_count = sum(
            char.isdigit()
            for char in username
        )


        if digit_count >= 7:

            score -= 5

            reasons.append(
                "⚠ Email username contains unusually many numbers"
            )


        else:

            reasons.append(
                "✔ Email username does not contain excessive numbers"
            )


        # =====================================================
        # 3. DOMAIN FORMAT - 10 POINTS
        # =====================================================

        if (

            '.' in domain

            and len(domain) >= 4

            and not domain.startswith('.')

            and not domain.endswith('.')

        ):

            score += 10

            reasons.append(
                "✔ Email domain format is valid"
            )


        else:

            score -= 10

            reasons.append(
                "❌ Invalid email domain"
            )


        # =====================================================
        # 4. TRUSTED PROVIDER - 15 POINTS
        # =====================================================

        trusted_domains = {

            "gmail.com",

            "outlook.com",

            "hotmail.com",

            "yahoo.com",

            "icloud.com",

            "proton.me",

            "protonmail.com"
        }


        if domain in trusted_domains:

            score += 15

            reasons.append(
                f"✔ Trusted email provider detected ({domain})"
            )


        else:

            reasons.append(
                "⚠ Email domain is not a commonly trusted provider"
            )


        # =====================================================
        # 5. MX RECORD CHECK - 25 POINTS
        # =====================================================

        try:

            mx_records = dns.resolver.resolve(

                domain,

                'MX',

                lifetime=5
            )


            if mx_records:

                score += 25

                reasons.append(
                    "✔ Domain has a valid mail (MX) record"
                )


        except Exception:

            score -= 20

            reasons.append(
                "❌ No valid mail (MX) record found"
            )


        # =====================================================
        # 6. DOMAIN AGE
        # =====================================================

        try:

            domain_age, domain_status = get_domain_age(
                "https://" + domain
            )


            if domain_age != "Not Available":

                try:

                    age_years = int(
                        domain_age.split()[0]
                    )


                    if domain in trusted_domains:

                        score += 15

                        reasons.append(
                            f"✔ Established major email provider domain ({domain_age})"
                        )

                    else:

                        if age_years >= 5:

                            score += 10

                            reasons.append(
                                f"✔ Established domain ({domain_age} old)"
                            )

                        elif age_years >= 2:

                            score += 7

                            reasons.append(
                                f"✔ Relatively established domain ({domain_age} old)"
                            )

                        elif age_years >= 1:

                            score += 3

                            reasons.append(
                                f"🟡 Relatively new domain ({domain_age} old)"
                            )

                        else:

                            score -= 10

                            reasons.append(
                                f"⚠ Very new domain ({domain_age} old)"
                            )

                except:

                    if domain in trusted_domains:
                        score += 15
                        reasons.append(
                            "✔ Established major email provider domain"
                        )
                    else:
                        reasons.append(
                            "⚠ Domain age could not be interpreted"
                        )

            else:

                if domain in trusted_domains:
                    score += 15
                    reasons.append(
                        "✔ Established major email provider domain"
                    )
                else:
                    reasons.append(
                        "ℹ Domain age information is not available"
                    )


        except Exception:

            reasons.append(
                "ℹ Domain age check could not be completed"
            )


        # =====================================================
        # 7. SUSPICIOUS USERNAME KEYWORDS
        # =====================================================

        suspicious_words = [

            "admin",

            "support",

            "security",

            "verify",

            "verification",

            "account",

            "payment",

            "bank",

            "billing",

            "login",

            "winner",

            "prize",

            "refund",

            "urgent",

            "alert"
        ]


        found_words = [

            word

            for word in suspicious_words

            if word in username
        ]


        if found_words:

            score -= 15

            reasons.append(
                "⚠ Suspicious keyword detected: "
                + ", ".join(found_words)
            )


        else:

            reasons.append(
                "✔ No suspicious keyword detected in username"
            )


        # =====================================================
        # 8. DISPOSABLE EMAIL CHECK
        # =====================================================

        disposable_domains = {

            "mailinator.com",

            "10minutemail.com",

            "tempmail.com",

            "guerrillamail.com",

            "yopmail.com",

            "temp-mail.org"
        }


        if domain in disposable_domains:

            score -= 25

            reasons.append(
                "❌ Disposable or temporary email domain detected"
            )


        # =====================================================
        # 9. SUSPICIOUS TLD CHECK
        # =====================================================

        suspicious_tlds = {

            ".xyz",

            ".top",

            ".click",

            ".work",

            ".zip",

            ".tk",

            ".ml",

            ".ga",

            ".cf"
        }


        if any(
            domain.endswith(tld)
            for tld in suspicious_tlds
        ):

            score -= 10

            reasons.append(
                "⚠ Domain uses a higher-risk TLD"
            )


        # =====================================================
        # 10. LIMITATION
        # =====================================================

        reasons.append(
            "ℹ Individual email account ownership or existence cannot be verified from the email address alone"
        )


        # =====================================================
        # FINAL SCORE
        # =====================================================

        score = max(
            0,
            min(score, 100)
        )


        # =====================================================
        # RISK CLASSIFICATION
        # =====================================================

        if score >= 85:

            status = "🟢 Safe Email"


        elif score >= 70:

            status = "🟡 Low Risk Email"


        elif score >= 45:

            status = "🟠 Medium Risk Email"


        else:

            status = "🔴 Suspicious Email"


        # =====================================================
        # RETURN
        # =====================================================

        return render_template(

            'email.html',

            email=email,

            score=score,

            status=status,

            reasons=reasons,

            domain_age=domain_age,

            domain_status=domain_status
        )


    return render_template(
        'email.html'
    )


# =========================================================
# QR ANALYSIS
# =========================================================

@app.route(
    "/qr",
    methods=["GET", "POST"]
)
def qr():

    if request.method == "POST":

        is_ajax = request.headers.get(
            "X-Requested-With"
        ) == "XMLHttpRequest"


        qr_data = request.form.get(
            "qr_data",
            ""
        ).strip()


        if not qr_data:

            if is_ajax:

                return jsonify(
                    success=False,
                    message="No QR data received"
                ), 400


            return "No QR data received"


        score, status, results, content_type, url_found = analyze_qr_content(
            qr_data
        )


        website_score = None

        website_status = None

        website_results = None


        if url_found:

            website_score, website_status, website_results = analyze_website(
                url_found
            )


        if is_ajax:

            return jsonify(

                success=True,

                qr_data=qr_data,

                content_type=content_type,

                score=score,

                status=status,

                results=results,

                website_score=website_score,

                website_status=website_status,

                website_results=website_results
            )


        return render_template(

            "qr.html",

            qr_data=qr_data,

            content_type=content_type,

            score=score,

            status=status,

            results=results,

            website_score=website_score,

            website_status=website_status,

            website_results=website_results
        )


    return render_template(
        "qr.html"
    )


# =========================================================
# IMAGE ANALYSIS ROUTE
# =========================================================

@app.route(
    "/image",
    methods=["GET", "POST"]
)
def image_analysis():

    if request.method == "POST":

        if 'image_file' not in request.files:

            return render_template(
                "image.html",
                error="No file uploaded"
            )


        file = request.files[
            'image_file'
        ]


        if file.filename == '':

            return render_template(
                "image.html",
                error="No file selected"
            )


        filename = secure_filename(
            file.filename
        )


        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        file.save(
            filepath
        )


        score, status, results, ai_percentage, edit_type, edit_explanation = analyze_image_forensics(
            filepath
        )


        try:

            os.remove(
                filepath
            )

        except Exception:

            pass


        return render_template(

            "image.html",

            score=score,

            status=status,

            results=results,

            ai_percentage=ai_percentage,

            edit_type=edit_type,

            edit_explanation=edit_explanation,

            filename=file.filename
        )


    return render_template(
        "image.html"
    )


# =========================================================
# VIDEO ANALYSIS
# =========================================================

@app.route(
    "/video",
    methods=["GET", "POST"]
)
def video_analysis():

    if request.method == "POST":

        if 'video_file' not in request.files:

            return render_template(
                "video.html",
                error="No file uploaded"
            )


        file = request.files[
            'video_file'
        ]


        if file.filename == '':

            return render_template(
                "video.html",
                error="No file selected"
            )


        filename = secure_filename(
            file.filename
        )


        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        file.save(
            filepath
        )


        score, status, results, ai_percentage, edit_type, edit_explanation = analyze_video_forensics(
            filepath
        )


        try:

            os.remove(
                filepath
            )

        except Exception:

            pass


        return render_template(

            "video.html",

            score=score,

            status=status,

            results=results,

            ai_percentage=ai_percentage,

            edit_type=edit_type,

            edit_explanation=edit_explanation,

            filename=file.filename
        )


    return render_template(
        "video.html"
    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False,
        port=5000
    )