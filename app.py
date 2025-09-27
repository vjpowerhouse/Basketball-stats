import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from typing import List, Dict, Any

st.set_page_config(page_title="Team Game Stats", page_icon="🏀", layout="wide")

# ---------- Config / Constants ----------

EVENT_TYPES = [
    "2PT_MADE","2PT_MISS","3PT_MADE","3PT_MISS","FT_MADE","FT_MISS",
    "OREB","DREB","AST","STL","BLK","TOV","PF","FD"
]

DEFAULT_QUARTERS = ["Q1","Q2","Q3","Q4","OT"]

# Landscape columns (left-to-right)
LANDSCAPE_BUTTON_COLS = [
    ("+2", "2PT_MADE"),
    ("2 Miss", "2PT_MISS"),
    ("+3", "3PT_MADE"),
    ("3 Miss", "3PT_MISS"),
    ("FT", "FT_MADE"),
    ("FT Miss", "FT_MISS"),
    ("OREB", "OREB"),
    ("DREB", "DREB"),
    ("AST", "AST"),
    ("STL", "STL"),
    ("BLK", "BLK"),
    ("TOV", "TOV"),
    ("PF", "PF"),
    ("FD", "FD"),
]
HEADER_REPEAT_EVERY = 8  # repeat column header row after this many players

# ---------- State ----------

def init_state():
    if "roster" not in st.session_state:
        # jersey # and name are both optional
        st.session_state.roster = pd.DataFrame(columns=["#", "Player"])
    if "events" not in st.session_state:
        st.session_state.events = pd.DataFrame(columns=[
            "game_id","datetime","quarter","clock",
            "player_display","player_name","#","number","event","notes"
        ])
    if "game_meta" not in st.session_state:
        st.session_state.game_meta = {
            "opponent": "",
            "date": datetime.today().date().isoformat(),
            "game_id": ""
        }
    if "quarters" not in st.session_state:
        st.session_state.quarters = DEFAULT_QUARTERS.copy()
    if "layout_mode" not in st.session_state:
        st.session_state.layout_mode = "Portrait"  # Portrait | Landscape
    if "last_undone" not in st.session_state:
        st.session_state.last_undone = None

init_state()

# ---------- Helpers ----------

def make_game_id(opponent: str, date_str: str) -> str:
    safe_opp = "".join([c for c in opponent if c.isalnum()])[:12].upper() or "OPP"
    safe_date = "".join(date_str.split("-"))
    return f"{safe_date}_{safe_opp}"

def player_display_from_row(row: pd.Series) -> str:
    num = str(row.get("#") or "").strip()
    name = str(row.get("Player") or "").strip()
    if num and name:
        return f"#{num} {name}"
    elif name:
        return name
    elif num:
        return f"#{num}"
    else:
        return ""   # completely blank if both empty

def pretty_name(rec: pd.Series) -> str:
    num = str(rec.get("#") or "").strip()
    name = str(rec.get("player_name") or "").strip()
    if num and name:
        return f"#{num} {name}"
    elif name:
        return name
    elif num:
        return f"#{num}"
    else:
        return ""   # blank

def add_event(game_id: str, quarter: str, clock: str,
              player_display: str, player_name: str, number: str,
              event: str, notes: str = ""):
    row = {
        "game_id": game_id,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "quarter": quarter,
        "clock": clock,
        "player_display": player_display,
        "player_name": player_name,
        "#": number,
        "number": number,
        "event": event,
        "notes": notes
    }
    st.session_state.events = pd.concat([st.session_state.events, pd.DataFrame([row])], ignore_index=True)

def undo_last_event():
    if st.session_state.events.empty:
        st.info("No events to undo.")
        st.session_state.last_undone = None
        return
    last_row = st.session_state.events.iloc[[-1]].to_dict(orient="records")[0]
    st.session_state.events = st.session_state.events.iloc[:-1].reset_index(drop=True)
    st.session_state.last_undone = last_row

def get_event_count(display: str, ev_code: str) -> int:
    if st.session_state.events.empty:
        return 0
    m = st.session_state.events
    return int(((m["player_display"] == display) & (m["event"] == ev_code)).sum())

def remove_last_for(display: str, ev_code: str) -> bool:
    """Remove the most recent event row for this player & event code."""
    if st.session_state.events.empty:
        return False
    df = st.session_state.events
    idxs = df.index[(df["player_display"] == display) & (df["event"] == ev_code)].tolist()
    if not idxs:
        return False
    last_idx = idxs[-1]
    st.session_state.events = df.drop(index=last_idx).reset_index(drop=True)
    return True

def compute_box(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "Player","MIN","PTS","FGM","FGA","FG%","3PM","3PA","3P%","FTM","FTA","FT%",
            "OREB","DREB","REB","AST","STL","BLK","TOV","PF"
        ])

    ev = events.copy()
    grp = ev.groupby(["#", "player_name", "player_display"], dropna=False)

    def pct(made, att):
        return np.where(att>0, np.round(100*made/att,1), np.nan)

    two_m = grp.apply(lambda g: (g["event"]=="2PT_MADE").sum()).rename("2PM")
    two_a = grp.apply(lambda g: ((g["event"]=="2PT_MADE") | (g["event"]=="2PT_MISS")).sum()).rename("2PA")
    thr_m = grp.apply(lambda g: (g["event"]=="3PT_MADE").sum()).rename("3PM")
    thr_a = grp.apply(lambda g: ((g["event"]=="3PT_MADE") | (g["event"]=="3PT_MISS")).sum()).rename("3PA")
    ft_m  = grp.apply(lambda g: (g["event"]=="FT_MADE").sum()).rename("FTM")
    ft_a  = grp.apply(lambda g: ((g["event"]=="FT_MADE") | (g["event"]=="FT_MISS")).sum()).rename("FTA")
    oreb  = grp.apply(lambda g: (g["event"]=="OREB").sum()).rename("OREB")
    dreb  = grp.apply(lambda g: (g["event"]=="DREB").sum()).rename("DREB")
    ast   = grp.apply(lambda g: (g["event"]=="AST").sum()).rename("AST")
    stl   = grp.apply(lambda g: (g["event"]=="STL").sum()).rename("STL")
    blk   = grp.apply(lambda g: (g["event"]=="BLK").sum()).rename("BLK")
    tov   = grp.apply(lambda g: (g["event"]=="TOV").sum()).rename("TOV")
    pf    = grp.apply(lambda g: (g["event"]=="PF").sum()).rename("PF")

    box = pd.concat([two_m, two_a, thr_m, thr_a, ft_m, ft_a, oreb, dreb, ast, stl, blk, tov, pf], axis=1).reset_index()

    box["FGM"] = box["2PM"] + box["3PM"]
    box["FGA"] = box["2PA"] + box["3PA"]
    box["PTS"] = box["2PM"]*2 + box["3PM"]*3 + box["FTM"]
    box["REB"] = box["OREB"] + box["DREB"]
    box["FG%"] = pct(box["FGM"], box["FGA"])
    box["3P%"] = pct(box["3PM"], box["3PA"])
    box["FT%"] = pct(box["FTM"], box["FTA"])
    box["MIN"] = np.nan

    box["Player"] = box.apply(pretty_name, axis=1)
    cols = ["Player","MIN","PTS","FGM","FGA","FG%","3PM","3PA","3P%","FTM","FTA","FT%",
            "OREB","DREB","REB","AST","STL","BLK","TOV","PF"]
    box = box[cols].sort_values(["PTS","REB","AST"], ascending=[False, False, False]).reset_index(drop=True)
    return box

def save_to_sqlite(db_path: str, roster: pd.DataFrame, events: pd.DataFrame, game_meta: Dict[str,Any]):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS players (
        number TEXT,
        name TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        opponent TEXT,
        date TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS events (
        game_id TEXT,
        datetime TEXT,
        quarter TEXT,
        clock TEXT,
        player_display TEXT,
        player_name TEXT,
        number TEXT,
        event TEXT,
        notes TEXT
    )""")
    # Replace players table contents with current roster snapshot
    conn.execute("DELETE FROM players")
    to_store = roster.rename(columns={"#":"number","Player":"name"})[["number","name"]].copy()
    to_store.to_sql("players", conn, if_exists="append", index=False)
    # Upsert game meta
    conn.execute("INSERT OR REPLACE INTO games (game_id, opponent, date) VALUES (?, ?, ?)",
                 (game_meta["game_id"], game_meta["opponent"], game_meta["date"]))
    # Append events
    events_out = events.drop(columns=["#"], errors="ignore").copy()
    events_out.to_sql("events", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

# ---------- Sidebar: Game Settings & Layout ----------

with st.sidebar:
    st.header("Game Settings")
    opponent = st.text_input("Opponent", value=st.session_state.game_meta.get("opponent",""))
    date_str = st.date_input("Date", value=pd.to_datetime(st.session_state.game_meta.get("date", datetime.today().date()))).isoformat()
    q_custom = st.text_input("Quarters (comma separated)", value=",".join(st.session_state.quarters))

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if st.button("Apply"):
            st.session_state.quarters = [q.strip() for q in q_custom.split(",") if q.strip()]
            gid = make_game_id(opponent, date_str)
            st.session_state.game_meta = {"opponent": opponent, "date": date_str, "game_id": gid}
            st.success(f"Game set: {gid}")
    with col_g2:
        st.session_state.layout_mode = st.radio("Layout", ["Portrait","Landscape"], horizontal=True, index=0)

# ---------- Tabs ----------

st.title("🏀 Team Game Stats — Courtside Logger")
tabs = st.tabs(["Game", "Roster", "Exports/DB"])

# --------- TAB: Game ---------
with tabs[0]:
    meta = st.session_state.game_meta
    st.write(f"**Game:** {meta.get('game_id','—')}  |  **Opponent:** {meta.get('opponent','—')}  |  **Date:** {meta.get('date','—')}  |  **Layout:** {st.session_state.layout_mode}")

    # Top controls shared
    top_c1, top_c2, top_c3, top_c4 = st.columns([1,1,2,1])
    quarter_sel = top_c1.selectbox("Quarter", st.session_state.quarters, index=0, key="quarter_main")
    clock_sel = top_c2.text_input("Clock (MM:SS)", value="", key="clock_main")

    if top_c4.button("↩️ Undo Last Event", use_container_width=True):
        undo_last_event()
        if st.session_state.last_undone:
            lu = st.session_state.last_undone
            who = lu.get("player_display") or ""
            evc = lu.get("event") or ""
            qtr = lu.get("quarter") or ""
            clk = lu.get("clock") or ""
            st.success(f"Undid: {who} — {evc} ({qtr} {clk})")

    # Clean roster view (skip completely blank rows)
    roster = st.session_state.roster.copy()
    def _is_blank_row(r):
        return not (str(r.get("#") or "").strip() or str(r.get("Player") or "").strip())
    if not roster.empty:
        roster = roster[~roster.apply(_is_blank_row, axis=1)].reset_index(drop=True)

    if st.session_state.layout_mode == "Portrait":
        st.subheader("Quick Log — One Tap")
        if roster.empty:
            st.info("Add players on the **Roster** tab to enable quick logging.")
        else:
            for idx, row in roster.iterrows():
                display = player_display_from_row(row)
                name = str(row.get("Player") or "").strip()
                num = str(row.get("#") or "").strip()

                if not display:
                    continue  # skip fully blank
                st.markdown(f"**{display}**")
                c1, c2, c3 = st.columns(3)
                if c1.button("+2", key=f"qbtn_{idx}_2"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "2PT_MADE")
                if c2.button("+3", key=f"qbtn_{idx}_3"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "3PT_MADE")
                if c3.button("FT", key=f"qbtn_{idx}_ft"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "FT_MADE")
                c4, c5, c6 = st.columns(3)
                if c4.button("OREB", key=f"qbtn_{idx}_oreb"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "OREB")
                if c5.button("AST", key=f"qbtn_{idx}_ast"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "AST")
                if c6.button("TOV", key=f"qbtn_{idx}_tov"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "TOV")
                c7, c8, c9 = st.columns(3)
                if c7.button("2 Miss", key=f"qbtn_{idx}_2miss"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "2PT_MISS")
                if c8.button("3 Miss", key=f"qbtn_{idx}_3miss"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "3PT_MISS")
                if c9.button("FT Miss", key=f"qbtn_{idx}_ftmiss"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "FT_MISS")
                c10, c11, c12 = st.columns(3)
                if c10.button("DREB", key=f"qbtn_{idx}_dreb"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "DREB")
                if c11.button("STL", key=f"qbtn_{idx}_stl"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "STL")
                if c12.button("BLK", key=f"qbtn_{idx}_blk"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "BLK")
                c13, c14 = st.columns(2)
                if c13.button("PF", key=f"qbtn_{idx}_pf"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "PF")
                if c14.button("FD", key=f"qbtn_{idx}_fd"):
                    add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, "FD")
                st.divider()

        st.subheader("Event Log")
        if st.session_state.events.empty:
            st.write("No events yet.")
        else:
            st.dataframe(st.session_state.events.sort_values("datetime", ascending=False), use_container_width=True, height=320)

    else:
        # LANDSCAPE grid with + / count / - stacked vertically in each cell
        st.subheader("Landscape Grid — Tap Actions")
        if roster.empty:
            st.info("Add players on the **Roster** tab to enable the landscape grid.")
        else:
            def header_row():
                cols = st.columns([2] + [1]*len(LANDSCAPE_BUTTON_COLS))
                cols[0].markdown("**Player**")
                for i, (label, _) in enumerate(LANDSCAPE_BUTTON_COLS, start=1):
                    cols[i].markdown(f"**{label}**")

            header_row()
            for ridx, row in roster.iterrows():
                # repeat headers every N rows to mimic frozen headings
                if ridx > 0 and ridx % HEADER_REPEAT_EVERY == 0:
                    st.markdown("---")
                    header_row()

                display = player_display_from_row(row)
                if not display:
                    continue  # skip fully blank
                name = str(row.get("Player") or "").strip()
                num = str(row.get("#") or "").strip()

                row_cols = st.columns([2] + [1]*len(LANDSCAPE_BUTTON_COLS))
                row_cols[0].markdown(f"**{display}**")

                for cidx, (label, ev_code) in enumerate(LANDSCAPE_BUTTON_COLS, start=1):
                    with row_cols[cidx]:
                        bp, bc, bm = st.columns([1,1,1])
                        # plus
                        if bp.button("+", key=f"plus_{ridx}_{ev_code}"):
                            add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, ev_code)
                        # count (center)
                        count = get_event_count(display, ev_code)
                        bc.markdown(f"<div style='text-align:center; font-weight:600;'>{count}</div>", unsafe_allow_html=True)
                        # minus
                        if bm.button("–", key=f"minus_{ridx}_{ev_code}"):
                            removed = remove_last_for(display, ev_code)
                            if not removed:
                                st.toast("Nothing to remove for this cell.", icon="⚠️")

        st.subheader("Event Log")
        if st.session_state.events.empty:
            st.write("No events yet.")
        else:
            st.dataframe(st.session_state.events.sort_values("datetime", ascending=False), use_container_width=True, height=320)

    # Box Score (common to both)
    st.subheader("Box Score (auto: makes & attempts)")
    box = compute_box(st.session_state.events)
    if box.empty:
        st.write("No stats yet.")
    else:
        st.dataframe(box, use_container_width=True, height=420)

# --------- TAB: Roster ---------
with tabs[1]:
    st.header("Roster Editor")
    st.caption("Enter **jersey #**, **player name**, or **both**. Leaving both blank will hide the row during logging.")
    if st.session_state.roster.empty:
        st.info("Start typing in the table below. Use ➕ to add rows.")
    edited = st.data_editor(
        st.session_state.roster if not st.session_state.roster.empty else pd.DataFrame(columns=["#", "Player"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "#": st.column_config.TextColumn("Jersey #", help="Optional: e.g., 3, 12, 25"),
            "Player": st.column_config.TextColumn("Player", help="Optional: name")
        },
        key="roster_editor_tab"
    )
    colR1, colR2 = st.columns(2)
    if colR1.button("✅ Save Roster", key="save_roster_tab"):
        # Keep as strings; blanks remain blanks
        st.session_state.roster = edited.astype(str)
        st.success(f"Roster saved ({len(edited)} players).")
    if colR2.button("🧹 Clear Roster", key="clear_roster_tab"):
        st.session_state.roster = pd.DataFrame(columns=["#","Player"])
        st.rerun()

# --------- TAB: Exports / DB ---------
with tabs[2]:
    st.header("Exports")
    colA, colB = st.columns(2)
    with colA:
        if not st.session_state.events.empty:
            csv = st.session_state.events.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Event Log (CSV)", csv,
                               file_name=f"{st.session_state.game_meta.get('game_id','game')}_events.csv",
                               mime="text/csv")
        else:
            st.write("No events to export yet.")
    with colB:
        box = compute_box(st.session_state.events)
        if not box.empty:
            csv2 = box.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Box Score (CSV)", csv2,
                               file_name=f"{st.session_state.game_meta.get('game_id','game')}_box.csv",
                               mime="text/csv")
        else:
            st.write("No box score yet.")

    st.divider()
    st.header("Persist to SQLite (optional)")
    use_db = st.checkbox("Enable SQLite save", value=False)
    db_path = st.text_input("Database file path", value="bball_stats.sqlite")
    if st.button("💾 Save to DB", disabled=not use_db):
        try:
            save_to_sqlite(
                db_path,
                st.session_state.roster.copy(),
                st.session_state.events.copy(),
                st.session_state.game_meta.copy()
            )
            st.success(f"Saved to {db_path}")
        except Exception as e:
            st.error(f"Error saving: {e}")