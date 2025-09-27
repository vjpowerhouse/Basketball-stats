import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from typing import List, Dict, Any

st.set_page_config(page_title="Team Game Stats", page_icon="🏀", layout="wide")

# ---------- Helpers ----------

EVENT_TYPES = [
    "2PT_MADE","2PT_MISS","3PT_MADE","3PT_MISS","FT_MADE","FT_MISS",
    "OREB","DREB","AST","STL","BLK","TOV","PF","FD"
]

DEFAULT_QUARTERS = ["Q1","Q2","Q3","Q4","OT"]

def init_state():
    if "roster" not in st.session_state:
        st.session_state.roster = pd.DataFrame(columns=["#","Player"])
    if "events" not in st.session_state:
        st.session_state.events = pd.DataFrame(columns=[
            "game_id","datetime","quarter","clock","player","#","event","notes"
        ])
    if "game_meta" not in st.session_state:
        st.session_state.game_meta = {"opponent":"", "date": datetime.today().date().isoformat(), "game_id": ""}
    if "quarters" not in st.session_state:
        st.session_state.quarters = DEFAULT_QUARTERS.copy()

def make_game_id(opponent: str, date_str: str) -> str:
    safe_opp = "".join([c for c in opponent if c.isalnum()])[:12].upper() or "OPP"
    safe_date = "".join(date_str.split("-"))
    return f"{safe_date}_{safe_opp}"

def add_event(game_id: str, quarter: str, clock: str, player: str, number: str, event: str, notes: str):
    row = {
        "game_id": game_id,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "quarter": quarter,
        "clock": clock,
        "player": player,
        "#": number,
        "event": event,
        "notes": notes
    }
    st.session_state.events = pd.concat([st.session_state.events, pd.DataFrame([row])], ignore_index=True)

def compute_box(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "#","Player","MIN","PTS","FGM","FGA","FG%","3PM","3PA","3P%","FTM","FTA","FT%",
            "OREB","DREB","REB","AST","STL","BLK","TOV","PF"
        ])

    ev = events.copy()
    grp = ev.groupby(["#","player"], dropna=False)

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
    box["MIN"] = np.nan  # placeholder

    box = box.rename(columns={"#":"Number","player":"Player"})
    cols = ["Number","Player","MIN","PTS","FGM","FGA","FG%","3PM","3PA","3P%","FTM","FTA","FT%",
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
        player TEXT,
        number TEXT,
        event TEXT,
        notes TEXT
    )""")
    conn.execute("DELETE FROM players")
    roster.to_sql("players", conn, if_exists="append", index=False)
    conn.execute("INSERT OR REPLACE INTO games (game_id, opponent, date) VALUES (?, ?, ?)", 
                 (game_meta["game_id"], game_meta["opponent"], game_meta["date"]))
    events.to_sql("events", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

def _validate_roster(df: pd.DataFrame) -> List[str]:
    errors = []
    if df.empty:
        errors.append("Add at least one player.")
        return errors
    if not set(["#","Player"]).issubset(df.columns):
        errors.append("Roster must have '#' and 'Player' columns.")
        return errors
    df["#"] = df["#"].astype(str).str.strip()
    df["Player"] = df["Player"].astype(str).str.strip()
    if (df["Player"]=="").any():
        errors.append("All players must have a name.")
    nums = df["#"][df["#"]!=""]
    if nums.duplicated().any():
        dups = nums[nums.duplicated()].unique().tolist()
        errors.append(f"Duplicate jersey numbers: {', '.join(dups)}")
    return errors

# ---------- UI ----------

init_state()

st.title("🏀 Team Game Stats — Courtside Logger")

with st.sidebar:
    st.header("1) Roster (In-App Editor)")
    st.caption("Add/edit players right here. Use ➕ to add rows; numbers are optional but should be unique.")
    edited = st.data_editor(
        st.session_state.roster if not st.session_state.roster.empty else pd.DataFrame(columns=["#", "Player"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "#": st.column_config.TextColumn("Jersey #", help="e.g., 3, 12, 25"),
            "Player": st.column_config.TextColumn("Player", required=True)
        },
        key="roster_editor"
    )
    colR1, colR2 = st.columns(2)
    if colR1.button("✅ Save Roster"):
        errs = _validate_roster(edited.copy())
        if errs:
            for e in errs:
                st.error(e)
        else:
            st.session_state.roster = edited.astype(str)
            st.success(f"Roster saved ({len(edited)} players).")
    if colR2.button("🧹 Clear Roster"):
        st.session_state.roster = pd.DataFrame(columns=["#","Player"])
        st.rerun()

    st.divider()
    st.header("2) Game Setup")
    opponent = st.text_input("Opponent", value=st.session_state.game_meta.get("opponent",""))
    date_str = st.date_input("Date", value=pd.to_datetime(st.session_state.game_meta.get("date", datetime.today().date()))).isoformat()
    q_custom = st.text_input("Quarters (comma separated)", value=",".join(st.session_state.quarters))
    if st.button("Apply Game Settings"):
        st.session_state.quarters = [q.strip() for q in q_custom.split(",") if q.strip()]
        gid = make_game_id(opponent, date_str)
        st.session_state.game_meta = {"opponent": opponent, "date": date_str, "game_id": gid}
        st.success(f"Game set: {gid}")

    st.divider()
    st.header("3) Save / Export")
    colA, colB = st.columns(2)
    with colA:
        if not st.session_state.events.empty:
            csv = st.session_state.events.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Event Log (CSV)", csv, file_name=f"{st.session_state.game_meta.get('game_id','game')}_events.csv", mime="text/csv")
    with colB:
        box = compute_box(st.session_state.events)
        if not box.empty:
            csv2 = box.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Box Score (CSV)", csv2, file_name=f"{st.session_state.game_meta.get('game_id','game')}_box.csv", mime="text/csv")

    st.caption("Tip: Use the quick buttons on the right for one-tap logging.")

# Main layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Live Event Logger")
    meta = st.session_state.game_meta
    st.write(f"**Game:** {meta.get('game_id','—')}  |  **Opponent:** {meta.get('opponent','—')}  |  **Date:** {meta.get('date','—')}")

    with st.form("add_event_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1,1,2])
        quarter = c1.selectbox("Quarter", st.session_state.quarters, index=0)
        clock = c2.text_input("Clock (MM:SS)", value="")
        roster = st.session_state.roster
        if roster.empty:
            st.info("Add a roster in the sidebar to start logging.")
            player = ""
            num = ""
        else:
            player = c3.selectbox("Player", roster["Player"].tolist())
            num = roster.loc[roster["Player"] == player, "#"].values[0] if player else ""
        c4, c5 = st.columns([2,1])
        event = c4.selectbox("Event", EVENT_TYPES, index=0)
        notes = c4.text_input("Notes (optional)", value="")
        submitted = st.form_submit_button("➕ Add Event")
        if submitted:
            if not meta.get("game_id"):
                st.error("Please set up the game in the sidebar.")
            elif not player:
                st.error("Please choose a player.")
            else:
                add_event(meta["game_id"], quarter, clock, player, num, event, notes)
                st.success(f"Added: {quarter} {clock or ''} — #{num} {player} — {event}")

    st.markdown("#### Event Log")
    if st.session_state.events.empty:
        st.write("No events yet.")
    else:
        st.dataframe(st.session_state.events.sort_values("datetime", ascending=False), use_container_width=True, height=300)

with right:
    st.subheader("Quick Log — One Tap")
    if st.session_state.roster.empty:
        st.info("Add players to enable quick logging.")
    else:
        meta = st.session_state.game_meta
        quarter = st.selectbox("Quarter (quick)", st.session_state.quarters, index=0, key="quick_qtr")
        clock = st.text_input("Clock (quick)", value="", key="quick_clock")
        for _, row in st.session_state.roster.iterrows():
            num = str(row["#"]); name = row["Player"]
            st.markdown(f"**#{num} {name}**")
            c1, c2, c3 = st.columns(3)
            if c1.button("+2", key=f"{num}_2"):
                add_event(meta.get("game_id",""), quarter, clock, name, num, "2PT_MADE", "")
            if c2.button("+3", key=f"{num}_3"):
                add_event(meta.get("game_id",""), quarter, clock, name, num, "3PT_MADE", "")
            if c3.button("FT", key=f"{num}_ft"):
                add_event(meta.get("game_id",""), quarter, clock, name, num, "FT_MADE", "")
            c4, c5, c6 = st.columns(3)
            if c4.button("REB", key=f"{num}_reb"):
                add_event(meta.get("game_id",""), quarter, clock, name, num, "DREB", "")
            if c5.button("AST", key=f"{num}_ast"):
                add_event(meta.get("game_id",""), quarter, clock, name, num, "AST", "")
            if c6.button("TOV", key=f"{num}_tov"):
                add_event(meta.get("game_id",""), quarter, clock, name, num, "TOV", "")
            st.divider()

    st.subheader("Box Score")
    box = compute_box(st.session_state.events)
    if box.empty:
        st.write("No stats yet.")
    else:
        st.dataframe(box, use_container_width=True, height=400)

    st.subheader("Persist to SQLite (optional)")
    use_db = st.checkbox("Enable SQLite save", value=False)
    db_path = st.text_input("Database file path", value="bball_stats.sqlite")
    if st.button("💾 Save to DB", disabled=not use_db):
        try:
            save_to_sqlite(db_path, st.session_state.roster.rename(columns={"Player":"name", "#":"number"}), st.session_state.events, st.session_state.game_meta)
            st.success(f"Saved to {db_path}")
        except Exception as e:
            st.error(f"Error saving: {e}")