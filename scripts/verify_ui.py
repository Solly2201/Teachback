"""Headless browser check of the faculty and student UI.

Unit tests prove the API behaves; this proves the *interface* a faculty member
actually sees does. It drives a real Chrome against the running dev server and
asserts on the rendered DOM, console errors and layout — in particular that the
lecture and topic Delete/Archive controls are genuinely visible and clickable,
since a control that exists only in the source is not a feature.

Prerequisites (started separately):
    backend :8000    python -m uvicorn app.main:app --port 8000
    frontend:5173    npm run dev        (or `npm run preview` on :4173)

Usage:
    python scripts/verify_ui.py [--url http://localhost:5173] [--headed]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((label, bool(ok), detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def make_driver(headed: bool) -> webdriver.Chrome:
    opts = Options()
    if not headed:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1440,900")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    return webdriver.Chrome(options=opts)


def body_text(driver) -> str:
    return driver.find_element(By.TAG_NAME, "body").text


def wait_text(driver, text: str, timeout: int = 30) -> bool:
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: text.lower() in body_text(d).lower())
        return True
    except TimeoutException:
        return False


def sign_in(driver, url: str, user: dict) -> None:
    """Set the role the way the app itself stores it, then load the route."""
    driver.get(url)
    driver.execute_script("localStorage.setItem('teachback_user', arguments[0])",
                          json.dumps(user))
    driver.execute_script("localStorage.removeItem('teachback_teacher_ctx')")


def console_errors(driver) -> list[str]:
    """Real page errors only — ignore favicon/vite/devtools noise."""
    out = []
    try:
        entries = driver.get_log("browser")
    except Exception:
        return out
    for e in entries:
        if e.get("level") != "SEVERE":
            continue
        msg = e.get("message", "")
        if "favicon" in msg or "/@vite/" in msg or "sourcemap" in msg.lower():
            continue
        out.append(msg)
    return out


def overflow(driver) -> int:
    return driver.execute_script(
        "return Math.max(0, document.documentElement.scrollWidth - "
        "document.documentElement.clientWidth)")


def api_get(url: str, path: str):
    import urllib.request
    with urllib.request.urlopen(f"{url}{path}", timeout=20) as r:
        return json.loads(r.read().decode())


def api_post(url: str, path: str, payload: dict):
    import urllib.request
    req = urllib.request.Request(
        f"{url}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def api_delete(url: str, path: str):
    import urllib.request
    req = urllib.request.Request(f"{url}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def card_titles(driver) -> list[str]:
    """The topic/lecture names currently rendered on the page."""
    return [e.text.strip() for e in driver.find_elements(By.CSS_SELECTOR, ".card .font-bold")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5173")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--shot", help="save a screenshot of the delete dialog here")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    driver = make_driver(args.headed)
    driver.set_page_load_timeout(60)
    try:
        # ------------------------------------------------- the sign-in path
        section("Role selection (the real entry point)")
        driver.get(url)
        driver.execute_script("localStorage.clear()")
        driver.get(url)
        check("role selection screen renders", wait_text(driver, "Continue as Teacher"))
        teacher_btn = next((b for b in driver.find_elements(By.TAG_NAME, "button")
                            if "Continue as Teacher" in b.text), None)
        check("the teacher entry button is clickable", teacher_btn is not None)
        if teacher_btn:
            teacher_btn.click()
            check("choosing Teacher lands on the faculty dashboard",
                  wait_text(driver, "Class Overview") or wait_text(driver, "Overview"))

        # ------------------------------------------------------ faculty list
        section("Faculty -> Lectures: is the Delete/Archive control visible?")
        # land on a subject that actually has a lecture, the way a teacher who
        # taught one would; an empty subject proves nothing about the control
        import urllib.request
        with urllib.request.urlopen(f"{url}/api/teachers", timeout=20) as r:
            teachers_api = json.loads(r.read().decode())
        target = None
        for t in teachers_api:
            for s in t["subjects"]:
                with urllib.request.urlopen(
                        f"{url}/api/lectures?subject_id={s['id']}", timeout=20) as r2:
                    if json.loads(r2.read().decode()):
                        target = (t["id"], s["id"], t["name"], s["name"])
                        break
            if target:
                break
        check("a seeded subject with at least one lecture exists", target is not None,
              f"{target[2]} / {target[3]}" if target else "none found")
        if target:
            driver.get(url)
            driver.execute_script(
                "localStorage.setItem('teachback_teacher_ctx', arguments[0])",
                json.dumps({"teacher_id": target[0], "subject_id": target[1]}))
        driver.get(f"{url}/lectures")
        check("the Lecture TeachBacks page renders", wait_text(driver, "Lecture TeachBacks"))
        time.sleep(2.5)  # let the lecture list resolve

        delete_buttons = [b for b in driver.find_elements(By.TAG_NAME, "button")
                          if b.text.strip().lower() == "delete"]
        visible = [b for b in delete_buttons if b.is_displayed()]
        check("a Delete control is rendered", len(delete_buttons) > 0,
              f"{len(delete_buttons)} found")
        check("the Delete control is VISIBLE to the user", len(visible) > 0,
              f"{len(visible)} visible")

        if visible:
            btn = visible[0]
            box = btn.rect
            check("Delete control has a real clickable area",
                  box["width"] >= 40 and box["height"] >= 20,
                  f"{int(box['width'])}x{int(box['height'])} px")
            unobstructed = driver.execute_script(
                "const el=arguments[0], r=el.getBoundingClientRect();"
                "const hit=document.elementFromPoint(r.left+r.width/2, r.top+r.height/2);"
                "return hit===el || el.contains(hit);", btn)
            check("Delete control is not covered by another element", unobstructed)
            check("Delete control has an accessible name",
                  bool((btn.get_attribute("title") or "").strip() or btn.text.strip()),
                  btn.get_attribute("title") or btn.text.strip())

        check("no horizontal overflow on the lecture list", overflow(driver) == 0,
              f"{overflow(driver)}px")

        # ------------------------------------------- confirmation dialog
        section("Delete confirmation dialog")
        if visible:
            visible[0].click()
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@role='dialog']")))
                dialog = driver.find_element(By.XPATH, "//div[@role='dialog']")
                text = dialog.text
                check("a confirmation dialog opens (not a one-click delete)",
                      dialog.is_displayed())
                check("the dialog explains what will happen",
                      len(text) > 80 and ("archiv" in text.lower() or "delete" in text.lower()),
                      text.split("\n")[0][:90])
                labels = [b.text.strip() for b in dialog.find_elements(By.TAG_NAME, "button")]
                check("the dialog offers Cancel and a confirm action",
                      any(l.lower() == "cancel" for l in labels) and len(labels) >= 2,
                      str(labels))
                if args.shot:
                    driver.save_screenshot(args.shot)
                    print(f"    (dialog screenshot saved to {args.shot})")
                cancel = next((b for b in dialog.find_elements(By.TAG_NAME, "button")
                               if b.text.strip().lower() == "cancel"), None)
                if cancel:
                    cancel.click()
                    time.sleep(1.2)
                    check("the dialog can be dismissed without deleting anything",
                          not driver.find_elements(By.XPATH, "//div[@role='dialog']"))
            except TimeoutException:
                check("a confirmation dialog opens (not a one-click delete)", False,
                      "no [role=dialog] appeared")

        # ------------------------------------------ topic management delete
        # The whole point of the action is that a teacher can remove a topic
        # from the real screen, so this drives it end to end on a probe topic
        # it creates and then removes — the database is left as it was found.
        section("Faculty -> Topic Management: delete a topic for real")
        PROBE = "UI probe topic"
        other_subject = None
        if target:
            for t in teachers_api:
                for sub in t["subjects"]:
                    if sub["id"] != target[1]:
                        other_subject = (t["id"], sub["id"])
                        break
                if other_subject:
                    break

            for stale in api_get(url, f"/api/topics?subject_id={target[1]}&include_archived=true"):
                if stale["name"] == PROBE:
                    api_delete(url, f"/api/topics/{stale['id']}")
            probe = api_post(url, "/api/topics", {
                "name": PROBE, "subject_id": target[1],
                "description": "Created by verify_ui.py and removed again.",
                "reference_explanation": "ref", "concepts": [], "relationships": [],
                "misconceptions": [], "activities": []})

            driver.get(f"{url}/topics")
            check("the Topic Management page renders", wait_text(driver, "Topic Management"))
            time.sleep(2.5)
            check("the new topic appears in the list", PROBE in card_titles(driver),
                  f"{len(card_titles(driver))} cards")

            del_btns = [b for b in driver.find_elements(By.TAG_NAME, "button")
                        if b.text.strip().lower() == "delete" and b.is_displayed()]
            check("a Delete control is VISIBLE on every topic card",
                  len(del_btns) >= len(card_titles(driver)) - 1,
                  f"{len(del_btns)} visible")
            # match the exact accessible name: the Edit control on the same
            # card also ends with the topic name
            probe_label = f"Delete topic {PROBE}"
            probe_btn = next(
                (b for b in del_btns
                 if (b.get_attribute("aria-label") or "") == probe_label), None)
            check("the probe topic has its own labelled Delete control",
                  probe_btn is not None,
                  probe_btn.get_attribute("aria-label") if probe_btn else "not found")

            if probe_btn:
                box = probe_btn.rect
                check("Delete control has a real clickable area",
                      box["width"] >= 40 and box["height"] >= 20,
                      f"{int(box['width'])}x{int(box['height'])} px")
                check("Delete control is not covered by another element",
                      driver.execute_script(
                          "const el=arguments[0], r=el.getBoundingClientRect();"
                          "const hit=document.elementFromPoint(r.left+r.width/2, r.top+r.height/2);"
                          "return hit===el || el.contains(hit);", probe_btn))
                check("no horizontal overflow on the topic list", overflow(driver) == 0,
                      f"{overflow(driver)}px")

                # --- Cancel must do nothing at all
                probe_btn.click()
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.XPATH, "//div[@role='dialog']")))
                    dialog = driver.find_element(By.XPATH, "//div[@role='dialog']")
                    check("clicking Delete opens a confirmation (not a one-click delete)",
                          dialog.is_displayed())
                    check("the dialog says the topic will be permanently deleted",
                          "permanently deleted" in dialog.text.lower(),
                          dialog.text.split("\n")[0][:90])
                    cancel = next((b for b in dialog.find_elements(By.TAG_NAME, "button")
                                   if b.text.strip().lower() == "cancel"), None)
                    if cancel:
                        cancel.click()
                        time.sleep(1.5)
                        check("Cancel dismisses the dialog",
                              not driver.find_elements(By.XPATH, "//div[@role='dialog']"))
                        check("Cancel changed nothing — the topic is still there",
                              any(t["name"] == PROBE for t in
                                  api_get(url, f"/api/topics?subject_id={target[1]}")))
                except TimeoutException:
                    check("clicking Delete opens a confirmation (not a one-click delete)",
                          False, "no [role=dialog] appeared")

                # --- confirm, and the topic must actually go
                probe_btn = next(
                    (b for b in driver.find_elements(By.TAG_NAME, "button")
                     if (b.get_attribute("aria-label") or "") == probe_label
                     and b.is_displayed()), None)
                if probe_btn:
                    probe_btn.click()
                    try:
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.XPATH, "//div[@role='dialog']")))
                        dialog = driver.find_element(By.XPATH, "//div[@role='dialog']")
                        confirm = next((b for b in dialog.find_elements(By.TAG_NAME, "button")
                                        if b.text.strip().lower().startswith("delete")), None)
                        check("the dialog offers an explicit confirm action",
                              confirm is not None)
                        if confirm:
                            confirm.click()
                            time.sleep(2.5)
                            check("the topic disappears from the list immediately",
                                  PROBE not in card_titles(driver))
                            check("the backend no longer has it",
                                  not any(t["name"] == PROBE for t in api_get(
                                      url, "/api/topics")))
                            driver.get(f"{url}/topics")
                            time.sleep(2.5)
                            check("refreshing the page does not bring it back",
                                  PROBE not in card_titles(driver))
                    except TimeoutException:
                        check("the confirmation dialog reopened for the delete", False,
                              "no [role=dialog] appeared")

            if other_subject:
                driver.get(url)
                driver.execute_script(
                    "localStorage.setItem('teachback_teacher_ctx', arguments[0])",
                    json.dumps({"teacher_id": other_subject[0],
                                "subject_id": other_subject[1]}))
                driver.get(f"{url}/topics")
                time.sleep(2.5)
                check("the other subject never shows the deleted topic",
                      PROBE not in card_titles(driver))

            # whatever happened above, do not leave the probe behind
            for stale in api_get(url, f"/api/topics?subject_id={target[1]}&include_archived=true"):
                if stale["name"] == PROBE:
                    api_delete(url, f"/api/topics/{stale['id']}")
            driver.get(url)
            driver.execute_script(
                "localStorage.setItem('teachback_teacher_ctx', arguments[0])",
                json.dumps({"teacher_id": target[0], "subject_id": target[1]}))

        # ------------------------------------------------- subject switcher
        section("Teacher / subject switcher actually refetches")
        driver.get(f"{url}/lectures")
        wait_text(driver, "Lecture TeachBacks")
        time.sleep(2)
        selects = [s for s in driver.find_elements(By.TAG_NAME, "select") if s.is_displayed()]
        check("teacher and subject selectors are present", len(selects) >= 2,
              f"{len(selects)} selects")
        if len(selects) >= 2:
            teacher_sel, subject_sel = Select(selects[0]), Select(selects[1])
            teachers = [o.text for o in teacher_sel.options]
            before = body_text(driver)
            if len(teachers) > 1:
                teacher_sel.select_by_index(1 if teacher_sel.first_selected_option.text == teachers[0] else 0)
                time.sleep(2.5)
                after = body_text(driver)
                check("switching teacher refetches the lecture list", before != after,
                      f"teachers: {teachers}")
                subjects = [o.text for o in
                            Select(driver.find_elements(By.TAG_NAME, "select")[1]).options]
                check("the subject list follows the selected teacher", len(subjects) >= 1,
                      f"subjects: {subjects}")

        # ------------------------------------------------- faculty routes
        section("Faculty routes render cleanly")
        for route in ["/", "/lectures", "/topics"]:
            driver.get(f"{url}{route}")
            time.sleep(2)
            text, errs, ov = body_text(driver), console_errors(driver), overflow(driver)
            check(f"faculty {route}", len(text.strip()) > 40 and not errs and ov == 0,
                  f"{len(text)} chars"
                  + (f" | console: {errs[0][:80]}" if errs else "")
                  + (f" | overflow {ov}px" if ov else ""))

        # ------------------------------------------------- student routes
        section("Student routes render cleanly")
        with urllib.request.urlopen(f"{url}/api/students", timeout=20) as r:
            students = json.loads(r.read().decode())
        demo = next((s for s in students if s.get("is_demo")), students[0])
        sign_in(driver, url, {"role": "student", "id": demo["id"], "name": demo["name"],
                              "program": demo.get("program", "")})
        for route in ["/", "/teachback", "/progress"]:
            driver.get(f"{url}{route}")
            time.sleep(2.2)
            text, errs, ov = body_text(driver), console_errors(driver), overflow(driver)
            check(f"student {route}", len(text.strip()) > 40 and not errs and ov == 0,
                  f"{len(text)} chars"
                  + (f" | console: {errs[0][:80]}" if errs else "")
                  + (f" | overflow {ov}px" if ov else ""))

        # ---------------------------------------- student-facing state wording
        section("Student-facing wording never accuses the student")
        driver.get(f"{url}/progress")
        time.sleep(2.2)
        text = body_text(driver)
        check("the phrase 'Not Trying' is not shown to students",
              "not trying" not in text.lower())

        # --------------------------------------------------- narrow desktop
        section("Layout at a narrow desktop width (1024px)")
        driver.set_window_size(1024, 800)
        for route in ["/", "/teachback", "/progress"]:
            driver.get(f"{url}{route}")
            time.sleep(1.8)
            check(f"no horizontal scrolling at 1024px on student {route}",
                  overflow(driver) == 0, f"{overflow(driver)}px")
        sign_in(driver, url, {"role": "teacher", "name": "Faculty",
                              "program": "Demo teacher access"})
        for route in ["/", "/lectures", "/topics"]:
            driver.get(f"{url}{route}")
            time.sleep(1.8)
            check(f"no horizontal scrolling at 1024px on faculty {route}",
                  overflow(driver) == 0, f"{overflow(driver)}px")
    finally:
        driver.quit()

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 74}")
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} UI checks passed")
    for label, _, detail in failed:
        print(f"  FAILED: {label} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
