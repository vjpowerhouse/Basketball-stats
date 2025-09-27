import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3, os, math
from typing import List, Dict, Any

st.set_page_config(page_title="Team Game Stats", page_icon="🏀", layout="wide")

# ---------- Config / Constants ----------

EVENT_TYPES = [
    "2PT_MADE","2PT_MISS","3PT_MADE","3PT_MISS","FT_MADE","FT_MISS",
    "OREB","DREB","AST","STL","BLK","TOV","PF","FD"
]

DEFAULT_QUARTERS = ["Q1","Q2","Q3","Q4","OT"]

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
HEADER_REPEAT_EVERY = 10
ROSTER_CACHE_PATH = "roster_cache.csv"

# ---------- State / Persistence ----------

def init_state():
    if "roster" not in st.session_state:
        st.session_state.roster = pd.DataFrame(columns=["#", "Player"])
        if os.path.exists(ROSTER_CACHE_PATH):
            try:
                cached = pd.read_csv(ROSTER_CACHE_PATH, dtype=str).fillna("")
                st.session_state.roster = cached[["#", "Player"]]
            except Exception:
                pass
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
        st.session_state.layout_mode = "Landscape"
    if "last_undone" not in st.session_state:
        st.session_state.last_undone = None

init_state()

# ---------- Helpers ----------

def clean(x: Any) -> str:
    """Normalize None/NaN/'None'/'nan' to '', and trim."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("none","nan"):
        return ""
    return s

def make_game_id(opponent: str, date_str: str) -> str:
    safe_opp = "".join([c for c in opponent if c.isalnum()])[:12].upper() or "OPP"
    safe_date = "".join(date_str.split("-"))
    return f"{safe_date}_{safe_opp}"

def player_display_from_row(row: pd.Series) -> str:
    num = clean(row.get("#"))
    name = clean(row.get("Player"))
    if num and name:
        return f"#{num} {name}"
    elif name:
        return name
    elif num:
        return f"#{num}"
    else:
        return ""   # completely blank if both empty

def pretty_name(rec: pd.Series) -> str:
    num = clean(rec.get("#"))
    name = clean(rec.get("player_name"))
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
        "player_display": clean(player_display),
        "player_name": clean(player_name),
        "#": clean(number),
        "number": clean(number),
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
    return int(((m["player_display"] == clean(display)) & (m["event"] == ev_code)).sum())

def remove_last_for(display: str, ev_code: str) -> bool:
    if st.session_state.events.empty:
        return False
    df = st.session_state.events
    key_disp = clean(display)
    idxs = df.index[(df["player_display"] == key_disp) & (df["event"] == ev_code)].tolist()
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
    conn.execute("DELETE FROM players")
    to_store = roster.rename(columns={"#":"number","Player":"name"})[["number","name"]].copy()
    to_store.to_sql("players", conn, if_exists="append", index=False)
    conn.execute("INSERT OR REPLACE INTO games (game_id, opponent, date) VALUES (?, ?, ?)",
                 (game_meta["game_id"], game_meta["opponent"], game_meta["date"]))
    events_out = events.drop(columns=["#"], errors="ignore").copy()
    events_out.to_sql("events", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

def save_roster_cache(df: pd.DataFrame):
    df.astype(str).fillna("").to_csv(ROSTER_CACHE_PATH, index=False)

# ---------- Sidebar ----------

with st.sidebar:
    st.header("Game Settings")
    opponent = st.text_input("Opponent", value=clean(st.session_state.game_meta.get("opponent","")))
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
        st.session_state.layout_mode = st.radio("Layout", ["Portrait","Landscape"], horizontal=True,
                                                index=(0 if st.session_state.layout_mode=="Portrait" else 1))

# ---------- CSS (fixed spacing, sticky headers, borders) ----------

st.markdown("""
<style>
.grid-wrap {
  overflow: auto;
  border: 2px solid #cfcfcf;
  border-radius: 8px;
}
.cell {
  border: 1px solid #dddddd;
  padding: 6px 8px;
  background: #fff;
}
.grid-header {
  position: sticky;
  top: 0;
  z-index: 5;
  background: #fff;
  border-bottom: 2px solid #cfcfcf;
}
.sticky-player {
  position: sticky;
  left: 0;
  z-index: 4;
  background: #fff;
  border-right: 2px solid #cfcfcf;
}
.cell-vert {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;               /* <-- adds space between +, count, – */
  min-height: 90px;       /* <-- ensures room so items don't overlap */
  justify-content: center;
}
.counter {
  text-align: center;
  font-weight: 700;
  padding: 2px 6px;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  min-width: 28px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Tabs ----------

st.title("🏀 Team Game Stats — Courtside Logger")
tabs = st.tabs(["Game", "Roster", "Exports/DB"])

# --------- TAB: Game ---------
with tabs[0]:
    meta = st.session_state.game_meta
    st.write(f"**Game:** {clean(meta.get('game_id')) or '—'}  |  **Opponent:** {clean(meta.get('opponent')) or '—'}  |  **Date:** {clean(meta.get('date')) or '—'}  |  **Layout:** {st.session_state.layout_mode}")

    top_c1, top_c2, top_c3, top_c4 = st.columns([1,1,2,1])
    quarter_sel = top_c1.selectbox("Quarter", st.session_state.quarters, index=0, key="quarter_main")
    clock_sel = top_c2.text_input("Clock (MM:SS)", value="", key="clock_main")
    if top_c4.button("↩️ Undo Last Event", use_container_width=True):
        undo_last_event()
        if st.session_state.last_undone:
            lu = st.session_state.last_undone
            who = clean(lu.get("player_display"))
            evc = clean(lu.get("event"))
            qtr = clean(lu.get("quarter"))
            clk = clean(lu.get("clock"))
            st.success(f"Undid: {who} — {evc} ({qtr} {clk})")

    # Clean roster (remove fully blank)
    roster = st.session_state.roster.copy()
    roster["#"] = roster["#"].map(clean)
    roster["Player"] = roster["Player"].map(clean)
    roster = roster[(roster["#"]!="") | (roster["Player"]!="")].reset_index(drop=True)

    if st.session_state.layout_mode == "Portrait":
        st.subheader("Quick Log — One Tap")
        if roster.empty:
            st.info("Add players on the **Roster** tab.")
        else:
            for idx, row in roster.iterrows():
                display = player_display_from_row(row)
                if not display:
                    continue
                name = clean(row.get("Player"))
                num = clean(row.get("#"))

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
            st.dataframe(st.session_state.events.sort_values("datetime", ascending=False),
                         use_container_width=True, height=320)

    else:
        # LANDSCAPE scoring sheet
        st.subheader("Scoring Sheet")
        if roster.empty:
            st.info("Add players on the **Roster** tab.")
        else:
            st.markdown("<div class='grid-wrap'>", unsafe_allow_html=True)

            # Header row (sticky)
            hdr_cols = st.columns([3] + [1]*len(LANDSCAPE_BUTTON_COLS))
            with hdr_cols[0]:
                st.markdown("<div class='cell grid-header sticky-player'><b>Player</b></div>", unsafe_allow_html=True)
            for i, (label, _) in enumerate(LANDSCAPE_BUTTON_COLS, start=1):
                with hdr_cols[i]:
                    st.markdown(f"<div class='cell grid-header' style='text-align:center'><b>{label}</b></div>",
                                unsafe_allow_html=True)

            # Player rows
            for ridx, row in roster.iterrows():
                if ridx > 0 and ridx % HEADER_REPEAT_EVERY == 0:
                    rpt_cols = st.columns([3] + [1]*len(LANDSCAPE_BUTTON_COLS))
                    with rpt_cols[0]:
                        st.markdown("<div class='cell grid-header sticky-player'><b>Player</b></div>", unsafe_allow_html=True)
                    for i, (label, _) in enumerate(LANDSCAPE_BUTTON_COLS, start=1):
                        with rpt_cols[i]:
                            st.markdown(f"<div class='cell grid-header' style='text-align:center'><b>{label}</b></div>",
                                        unsafe_allow_html=True)

                display = player_display_from_row(row)
                if not display:
                    continue
                name = clean(row.get("Player"))
                num = clean(row.get("#"))

                row_cols = st.columns([3] + [1]*len(LANDSCAPE_BUTTON_COLS))
                with row_cols[0]:
                    st.markdown(f"<div class='cell sticky-player'><b>{display}</b></div>", unsafe_allow_html=True)

                for cidx, (label, ev_code) in enumerate(LANDSCAPE_BUTTON_COLS, start=1):
                    with row_cols[cidx]:
                        st.markdown("<div class='cell'>", unsafe_allow_html=True)
                        st.markdown("<div class='cell-vert'>", unsafe_allow_html=True)

                        c_plus = st.button("+", key=f"plus_{ridx}_{ev_code}", use_container_width=True)
                        cnt = get_event_count(display, ev_code)
                        st.markdown(f"<div class='counter'>{cnt}</div>", unsafe_allow_html=True)
                        c_minus = st.button("–", key=f"minus_{ridx}_{ev_code}", use_container_width=True)

                        if c_plus:
                            add_event(meta.get("game_id",""), quarter_sel, clock_sel, display, name, num, ev_code)
                        if c_minus:
                            removed = remove_last_for(display, ev_code)
                            if not removed:
                                st.toast("Nothing to remove for this stat.", icon="⚠️")

                        st.markdown("</div>", unsafe_allow_html=True)  # end cell-vert
                        st.markdown("</div>", unsafe_allow_html=True)  # end cell

            st.markdown("</div>", unsafe_allow_html=True)  # end grid-wrap

        st.subheader("Event Log")
        if st.session_state.events.empty:
            st.write("No events yet.")
        else:
            st.dataframe(st.session_state.events.sort_values("datetime", ascending=False),
                         use_container_width=True, height=320)

    # Box Score
    st.subheader("Box Score (makes & attempts)")
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
            "#": st.column_config.TextColumn("Jersey #", help="Optional"),
            "Player": st.column_config.TextColumn("Player", help="Optional")
        },
        key="roster_editor_tab"
    )
    colR1, colR2, colR3 = st.columns(3)
    if colR1.button("✅ Save Roster", key="save_roster_tab"):
        st.session_state.roster = edited.copy()
        st.session_state.roster["#"] = st.session_state.roster["#"].map(clean)
        st.session_state.roster["Player"] = st.session_state.roster["Player"].map(clean)
        save_roster_cache(st.session_state.roster)  # persist last roster
        st.success(f"Roster saved ({len(st.session_state.roster)} players).")
    if colR2.button("🧹 Clear Roster", key="clear_roster_tab"):
        st.session_state.roster = pd.DataFrame(columns=["#","Player"])
        save_roster_cache(st.session_state.roster)
        st.rerun()
    if colR3.download_button("⬇️ Download Roster CSV",
                             st.session_state.roster.to_csv(index=False).encode("utf-8"),
                             file_name="roster.csv",
                             mime="text/csv"):
        pass

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