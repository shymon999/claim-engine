"""
CLAIM ENGINE v2.0 — Streamlit Web Edition
Deploy free on Streamlit Community Cloud.
"""
import streamlit as st
import pandas as pd
import os, io
from datetime import datetime, timedelta
from collections import Counter
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

st.set_page_config(page_title="Claim Engine v2.0", page_icon="🎯", layout="wide")

Base = declarative_base()

# ============================================================================
# MODELS (identical to desktop)
# ============================================================================

class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    display_name = Column(String(100))
    handlers = relationship("Handler", back_populates="team", cascade="all, delete-orphan")
    rules = relationship("Rule", back_populates="team", cascade="all, delete-orphan")

class Handler(Base):
    __tablename__ = 'handlers'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    riskonnect_id = Column(String(50), unique=True)
    team_name = Column(String(50))
    team_id = Column(Integer, ForeignKey('teams.id'))
    backup_handler_id = Column(Integer, ForeignKey('handlers.id'), nullable=True)
    is_present = Column(Boolean, default=True)
    team = relationship("Team", back_populates="handlers")
    backup_handler = relationship("Handler", remote_side=[id], foreign_keys=[backup_handler_id])

class Rule(Base):
    __tablename__ = 'rules'
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    priority = Column(Integer, default=50)
    description = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True)
    countries = Column(Text, nullable=True)
    divisions = Column(Text, nullable=True)
    claim_sub_types = Column(Text, nullable=True)
    customer_contains = Column(String(200), nullable=True)
    min_amount = Column(Float, nullable=True)
    max_amount = Column(Float, nullable=True)
    handler_ids = Column(Text, nullable=True)
    backup_handler_ids = Column(Text, nullable=True)
    output_team_name = Column(String(100), nullable=True)
    output_assigned_name = Column(String(100), nullable=True)
    team = relationship("Team", back_populates="rules")

class VIPCustomer(Base):
    __tablename__ = 'vip_customers'
    id = Column(Integer, primary_key=True)
    customer_name = Column(String(200))
    country = Column(String(100), nullable=True)
    handler_id = Column(Integer, ForeignKey('handlers.id'))
    min_amount = Column(Float, default=0)
    max_amount = Column(Float, default=999999)
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=10)
    handler = relationship("Handler")

class SpecialCustomer(Base):
    __tablename__ = 'special_customers'
    id = Column(Integer, primary_key=True)
    customer_name = Column(String(200))
    handler_ids = Column(Text)
    is_active = Column(Boolean, default=True)

class SchenkerConfig(Base):
    __tablename__ = 'schenker_config'
    id = Column(Integer, primary_key=True)
    country = Column(String(100))
    division = Column(String(100), default='all')
    merge_month = Column(Integer)
    is_active = Column(Boolean, default=True)

class ClaimSubType(Base):
    __tablename__ = 'claim_sub_types'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
    category = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)

class History(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    claim_number = Column(String(50))
    country = Column(String(100))
    division = Column(String(100))
    claimant = Column(String(200))
    amount = Column(Float)
    handler_name = Column(String(100))
    handler_rid = Column(String(50))
    team_name = Column(String(100))
    reason = Column(Text)


# ============================================================================
# UTILITIES
# ============================================================================

DIVISIONS = ['Road', 'A&S', 'Solutions', 'XPress', 'Contract Logistics']
TEAM_NAMES = ['CHC Global', 'CHC Nordic', 'CHC Bucharest', 'CHC Doc Team',
              'Team Schenker Legacy', 'Claims Schenker Legacy']
COUNTRIES_NORDIC = ['Denmark', 'Sweden', 'Norway', 'Finland', 'Estonia', 'Lithuania', 'Latvia']

def normalize_division(div):
    if not div: return ''
    div = str(div).strip()
    low = div.lower().replace(' ', '')
    if 'air' in div.lower() and 'sea' in div.lower(): return 'A&S'
    if low in ('a&s', 'as', 'airsea', 'air&sea'): return 'A&S'
    if 'xpress' in div.lower(): return 'XPress'
    if 'solution' in div.lower(): return 'Solutions'
    if 'road' in div.lower(): return 'Road'
    if 'contract' in div.lower() and 'logistic' in div.lower(): return 'Contract Logistics'
    return div

def handler_name_by_id(session, hid):
    h = session.get(Handler, int(hid))
    return h.name if h else '?'

def handler_names_from_ids(session, ids_str):
    if not ids_str: return ''
    names = []
    for hid in ids_str.split(','):
        if hid.strip().isdigit():
            h = session.get(Handler, int(hid.strip()))
            if h: names.append(h.name)
    return ', '.join(names)

def all_handlers_dict(session):
    """Returns {display_str: handler_id} for all handlers."""
    d = {}
    for h in session.query(Handler).order_by(Handler.team_name, Handler.name).all():
        d[f"{h.name} [{h.team_name}]"] = h.id
    return d


# ============================================================================
# CLAIM PROCESSOR (identical logic to desktop)
# ============================================================================

class ClaimProcessor:
    def __init__(self, session, team):
        self.session = session
        self.team = team
        self.load_counter = Counter()
        self.today = datetime.now()

    def process_dataframe(self, df):
        results = []
        for _, row in df.iterrows():
            handler, team_name, rid, reason = self._assign_claim(row)
            self._log_history(row, handler, team_name, rid, reason)
            results.append(self._build_output(row, handler, team_name, rid, reason))
        self.session.commit()
        return pd.DataFrame(results)

    def get_stats(self):
        return dict(self.load_counter)

    def _assign_claim(self, row):
        shipment = str(row.get('Shipment number', '')).strip()
        country = str(row.get('DSV Country (Lookup)', '')).strip()
        division = normalize_division(str(row.get('DSV Division (Lookup)', '')).strip())
        claimant = str(row.get('Claimant Name', '')).strip()
        sub_type = str(row.get('Claim Sub-Type', '')).strip()

        dol = row.get('Date of Loss')
        if pd.notna(dol):
            if isinstance(dol, str):
                try: dol = pd.to_datetime(dol, dayfirst=True)
                except: dol = None
        else:
            dol = None

        claim_amt = self._safe_float(row.get('Claim amount EUR', 0))
        liability = self._safe_float(row.get('Total liability EUR', 0))
        eff_amt = min(claim_amt, liability) if claim_amt > 0 and liability > 0 else max(claim_amt, liability)

        # STEP 0: Schenker
        sr = self._check_schenker(shipment, country, division, dol)
        if sr: return sr

        # STEP 1: Special Customers
        for sc in self.session.query(SpecialCustomer).filter_by(is_active=True).all():
            if sc.customer_name.lower() in claimant.lower():
                hids = [int(x) for x in sc.handler_ids.split(',') if x.strip().isdigit()]
                handlers = [self.session.get(Handler, hid) for hid in hids]
                handlers = [h for h in handlers if h and h.is_present]
                h = self._pick(handlers)
                if h: return h, h.team_name or 'CHC Global', h.riskonnect_id, f'Special: {sc.customer_name}'

        # STEP 2: VIP Customers
        for vip in self.session.query(VIPCustomer).filter_by(is_active=True).order_by(VIPCustomer.priority).all():
            if vip.customer_name.lower() not in claimant.lower(): continue
            if vip.country and vip.country.lower() != country.lower(): continue
            if not (vip.min_amount <= eff_amt < vip.max_amount): continue
            h = self.session.get(Handler, vip.handler_id)
            if h and h.is_present:
                self.load_counter[h.id] += 1
                return h, h.team_name or 'CHC Nordic', h.riskonnect_id, f'VIP: {vip.customer_name} ({eff_amt:.0f} EUR)'
            elif h and h.backup_handler and h.backup_handler.is_present:
                bh = h.backup_handler
                self.load_counter[bh.id] += 1
                return bh, bh.team_name or 'CHC Nordic', bh.riskonnect_id, f'VIP Backup: {vip.customer_name}'

        # STEP 3: Rules
        rules = (self.session.query(Rule).filter(Rule.team_id == self.team.id, Rule.is_active == True)
                 .order_by(Rule.priority, Rule.id).all())
        for rule in rules:
            if not self._rule_matches(rule, country, division, sub_type, claimant, eff_amt): continue
            if rule.output_assigned_name == '#N/A':
                return None, rule.output_team_name or self.team.display_name, '#N/A', f'Rule: {rule.description or "override"}'
            handlers = self._get_rule_handlers(rule)
            h = self._pick(handlers)
            if h:
                tn = rule.output_team_name or h.team_name or self.team.display_name
                amt_info = f' ({eff_amt:.0f} EUR)' if rule.min_amount is not None else ''
                return h, tn, h.riskonnect_id, f'Rule: {rule.description or ""}{amt_info}'

        return None, '', '#N/A', 'No matching rule'

    def _check_schenker(self, shipment, country, division, dol):
        if not shipment or '-' in shipment: return None
        configs = self.session.query(SchenkerConfig).filter_by(is_active=True).all()
        merge_map = {}
        for cfg in configs:
            merge_map.setdefault(cfg.country, []).append((cfg.division, cfg.merge_month))
        if country not in merge_map:
            if dol and dol.year >= 2026: return None
            return None, 'Claims Schenker Legacy', '#N/A', f'Schenker: {country} not in config'
        if not dol:
            return None, 'Claims Schenker Legacy', '#N/A', 'Schenker: no DoL'
        if dol.year >= 2026: return None
        if dol.year == 2025:
            for div_cfg, mm in merge_map[country]:
                if div_cfg == 'all' or normalize_division(div_cfg) == division:
                    if dol >= datetime(2025, mm, 1): return None
            return None, 'Claims Schenker Legacy', '#N/A', f'Schenker Legacy: {country} DoL 2025'
        return None, 'Claims Schenker Legacy', '#N/A', 'Schenker Legacy: DoL < 2025'

    def _rule_matches(self, rule, country, division, sub_type, claimant, eff_amt):
        if rule.countries:
            if not any(c.strip().lower() == country.lower() for c in rule.countries.split(',')): return False
        if rule.divisions:
            if division not in [normalize_division(d.strip()) for d in rule.divisions.split(',')]: return False
        if rule.claim_sub_types:
            if not any(s.strip().lower() in sub_type.lower() for s in rule.claim_sub_types.split(',')): return False
        if rule.customer_contains:
            if rule.customer_contains.lower() not in claimant.lower(): return False
        if rule.min_amount is not None and eff_amt < rule.min_amount: return False
        if rule.max_amount is not None and eff_amt >= rule.max_amount: return False
        return True

    def _get_rule_handlers(self, rule):
        handlers = []
        if rule.handler_ids:
            for hid in rule.handler_ids.split(','):
                if hid.strip().isdigit():
                    h = self.session.get(Handler, int(hid))
                    if h and h.is_present: handlers.append(h)
        if not handlers and rule.backup_handler_ids:
            for hid in rule.backup_handler_ids.split(','):
                if hid.strip().isdigit():
                    h = self.session.get(Handler, int(hid))
                    if h and h.is_present: handlers.append(h)
        if not handlers and rule.handler_ids:
            for hid in rule.handler_ids.split(','):
                if hid.strip().isdigit():
                    h = self.session.get(Handler, int(hid))
                    if h and h.backup_handler and h.backup_handler.is_present:
                        handlers.append(h.backup_handler)
        return handlers

    def _pick(self, handlers):
        if not handlers: return None
        sel = min(handlers, key=lambda h: self.load_counter.get(h.id, 0))
        self.load_counter[sel.id] = self.load_counter.get(sel.id, 0) + 1
        return sel

    def _safe_float(self, v):
        try: return float(v) if pd.notna(v) else 0.0
        except: return 0.0

    def _build_output(self, row, handler, team_name, rid, reason):
        r = row.copy()
        if 'Claim: Claim Number' in r.index:
            r = r.rename({'Claim: Claim Number': 'Claim Import ID'})
        r['Assigned Name'] = rid or '#N/A'
        r['Claim Handler'] = handler.name if handler else ''
        r['Team Name'] = team_name
        r['Assignment Reason'] = reason
        r['Internal Status'] = 'Awaiting own process'
        r['Recovery Status'] = 'Awaiting own process'
        r['Initial assignment'] = self.today.strftime('%d.%m.%Y')
        if str(row.get('Status', '')).strip().lower() == 'new':
            r['Status'] = 'Assigned'
        dol = row.get('Date of Loss')
        if pd.notna(dol):
            try:
                if isinstance(dol, str): dol = pd.to_datetime(dol, dayfirst=True)
                r['Timebar date client'] = (dol + timedelta(days=365)).strftime('%d.%m.%Y')
            except: pass
        return r

    def _log_history(self, row, handler, team_name, rid, reason):
        self.session.add(History(
            claim_number=str(row.get('Claim: Claim Number', row.get('Claim Import ID', ''))),
            country=str(row.get('DSV Country (Lookup)', '')),
            division=str(row.get('DSV Division (Lookup)', '')),
            claimant=str(row.get('Claimant Name', '')),
            amount=self._safe_float(row.get('Claim amount EUR', 0)),
            handler_name=handler.name if handler else '',
            handler_rid=rid or '', team_name=team_name or '', reason=reason
        ))


# ============================================================================
# DATABASE INIT + SEED
# ============================================================================

@st.cache_resource
def get_session():
    # Turso cloud DB (when deployed) or local SQLite (for development)
    turso_url = os.environ.get("TURSO_DATABASE_URL") or st.secrets.get("TURSO_DATABASE_URL", "")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN") or st.secrets.get("TURSO_AUTH_TOKEN", "")

    if turso_url and turso_token:
        # Remote Turso — persistent cloud database
        engine = create_engine(
            f"sqlite+libsql://{turso_url.replace('libsql://', '').replace('https://', '')}?secure=true",
            connect_args={"auth_token": turso_token},
        )
    else:
        # Local SQLite — for development / testing
        os.makedirs('data', exist_ok=True)
        engine = create_engine('sqlite:///data/claim_engine_v2.db', connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    if session.query(Team).count() > 0:
        return session

    # Teams
    teams = {}
    for name, display in [('Global', 'CHC Global'), ('Nordic', 'CHC Nordic'),
                           ('Bucharest', 'CHC Bucharest'), ('Doc', 'CHC Doc Team')]:
        t = Team(name=name, display_name=display)
        session.add(t); session.flush(); teams[name] = t

    # Handlers
    hd = {}
    for name, rid, tname, tkey in [
        ('Agata Więckowska', '005Ts000002590X', 'CHC Global', 'Global'),
        ('Hubert Zych', '005Ts00000GGV9tIAH', 'CHC Global', 'Global'),
        ('Justyna Klepczyńska-Buczek', '005Ts0000025CY1', 'CHC Global', 'Global'),
        ('Katarzyna Miszczak', '005Ts0000025D4L', 'CHC Global', 'Global'),
        ('Łukasz Twarowski', '005Ts00000AIxIPIA1', 'CHC Global', 'Global'),
        ('Oliwia Hagowska', '005Ts000001AlgE', 'CHC Global', 'Global'),
        ('Antoni Jełowicki', '005Vk00000FIXPvIAP', 'CHC Global', 'Global'),
        ('Razvan Poenaru', '005Ts00000J4GhZIAV', 'CHC Bucharest', 'Bucharest'),
        ('Kacper Tabiszewski', '005Ts000002X82V', 'CHC Doc Team', 'Doc'),
        ('Robert Kaczmarczyk', '005Vk00000FyPXa', 'CHC Nordic', 'Nordic'),
        ('Karolina Rygorowicz', '005Ts0000025D3n', 'CHC Nordic', 'Nordic'),
        ('Jakub Kowalski', '005Ts000001bFM5', 'CHC Nordic', 'Nordic'),
        ('Weronika Kruszewska', '005Ts0000025Bbc', 'CHC Nordic', 'Nordic'),
        ('Amelia Falk', '005Ts00000Dn3y9', 'CHC Nordic', 'Nordic'),
        ('Tomasz Wasil', '005Ts00000L4ib3', 'CHC Nordic', 'Nordic'),
        ('Michał Sztorc', '005Ts0000025EPZ', 'CHC Nordic', 'Nordic'),
        ('Szymon Michonski', '005Ts0000025Bbp', 'CHC Doc Team', 'Doc'),
        ('Angelic Arellano', '005Ts00000259QA', 'CHC Doc Team', 'Doc'),
        ('Arianne Dimalanta', '005Ts00000259UP', 'CHC Doc Team', 'Doc'),
        ('Chris-Ann Bautista', '005Ts000006oQdN', 'CHC Doc Team', 'Doc'),
    ]:
        h = Handler(name=name, riskonnect_id=rid, team_name=tname, team_id=teams[tkey].id)
        session.add(h); session.flush(); hd[name] = h

    # Backup
    hd['Weronika Kruszewska'].backup_handler_id = hd['Jakub Kowalski'].id

    # Sub-types
    for name, cat in [
        ('Damage', 'damage'), ('Water Damage', 'damage'), ('Hidden Damage', 'damage'),
        ('Temperature Damage', 'damage'), ('Contamination', 'damage'),
        ('Total Missing', 'manco'), ('Partial Missing', 'manco'),
        ('Delay', 'other'), ('Theft', 'other'), ('Errors & Omissions', 'other'),
        ('Truck Accident', 'other'), ('General Average', 'other'),
    ]:
        session.add(ClaimSubType(name=name, category=cat))

    # Special customers
    j_o = f"{hd['Justyna Klepczyńska-Buczek'].id},{hd['Oliwia Hagowska'].id}"
    for c in ['Abbott', 'Adidas', 'Autoliv', 'HP', 'Estee Lauder', 'WD - WESTERN DIGITAL', 'Burberry', 'SATAIR']:
        session.add(SpecialCustomer(customer_name=c, handler_ids=j_o))

    # Schenker config
    for country, div, month in [
        ('USA', 'all', 8), ('Denmark', 'all', 8),
        ('UK', 'all', 9), ('Switzerland', 'all', 9), ('Netherlands', 'A&S', 9),
        ('Norway', 'A&S', 9), ('Chile', 'A&S', 9), ('Chile', 'Road', 9), ('Panama', 'A&S', 9),
        ('Ireland', 'all', 10), ('Spain', 'A&S', 10), ('Spain', 'Contract Logistics', 10),
        ('Portugal', 'A&S', 10), ('Myanmar', 'A&S', 10), ('Myanmar', 'Contract Logistics', 10), ('Laos', 'A&S', 10),
        ('Belgium', 'all', 11), ('Hong Kong', 'all', 11), ('South Africa', 'all', 11),
        ('Bangladesh', 'all', 11), ('Norway', 'Road', 11), ('Luxembourg', 'A&S', 11),
        ('Luxembourg', 'Road', 11), ('Peru', 'A&S', 11), ('Peru', 'Contract Logistics', 11), ('Mozambique', 'A&S', 11),
        ('Italy', 'all', 12), ('Finland', 'all', 12), ('Taiwan', 'all', 12),
        ('Philippines', 'all', 12), ('Dubai', 'all', 12), ('Estonia', 'all', 12),
        ('Egypt', 'all', 12), ('Croatia', 'all', 12), ('Netherlands', 'Road', 12),
        ('Netherlands', 'Contract Logistics', 12), ('Greece', 'A&S', 12), ('Puerto Rico', 'A&S', 12),
    ]:
        session.add(SchenkerConfig(country=country, division=div, merge_month=month))

    razvan = hd['Razvan Poenaru']

    # Global rules from CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rules_full.csv')
    ft_config = {
        ('Belgium', 'Road'): (200, 500), ('Czech', 'Road'): (200, 500),
        ('France', 'Road'): (0, 200), ('Ireland', 'Solutions'): (0, 200),
        ('Ireland', 'Road'): (0, 200), ('Netherlands', 'Road'): (200, 500),
        ('Portugal', 'Road'): (0, 200), ('Spain', 'Road'): (0, 200),
        ('UK', 'Road'): (200, 500),
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            country = str(row['Country']).strip()
            divs = ','.join(normalize_division(d.strip()) for d in str(row['Division']).split(','))
            main_str = str(row['Main Handler(s)']).strip()
            alt_str = str(row.get('Alternative', '')).strip()
            main_ids = [str(hd[n].id) for n in hd if n in main_str]
            alt_ids = [str(hd[n].id) for n in hd if alt_str and alt_str != '-' and n in alt_str]
            session.add(Rule(team_id=teams['Global'].id, priority=50,
                description=f"{country} {divs}", countries=country, divisions=divs,
                handler_ids=','.join(main_ids) or None,
                backup_handler_ids=','.join(alt_ids) or None))
            key = (country, divs.split(',')[0])
            if key in ft_config:
                ft_min, ft_max = ft_config[key]
                session.add(Rule(team_id=teams['Global'].id, priority=20,
                    description=f"FT: {country} {divs} {ft_min}-{ft_max}EUR",
                    countries=country, divisions=divs,
                    min_amount=ft_min if ft_min > 0 else None, max_amount=ft_max,
                    handler_ids=str(razvan.id), output_team_name='CHC Bucharest'))

    # XPress
    session.add(Rule(team_id=teams['Global'].id, priority=5,
        description='XPress all countries', divisions='XPress',
        handler_ids=f"{hd['Oliwia Hagowska'].id},{hd['Łukasz Twarowski'].id}"))

    # Nordic rules
    session.add(Rule(team_id=teams['Nordic'].id, priority=10,
        description='Low Value <200 (DK,EE,LT,LV,NO,FI)',
        countries='Denmark,Estonia,Lithuania,Latvia,Norway,Finland', max_amount=200,
        handler_ids=str(hd['Angelic Arellano'].id), output_team_name='CHC Doc Team'))
    session.add(Rule(team_id=teams['Nordic'].id, priority=10,
        description='SE Low Value <200', countries='Sweden', max_amount=200,
        handler_ids=str(hd['Arianne Dimalanta'].id), output_team_name='CHC Doc Team'))
    session.add(Rule(team_id=teams['Nordic'].id, priority=15,
        description='Fast Track 200-500 (DK,EE,LT,LV,NO,FI)',
        countries='Denmark,Estonia,Lithuania,Latvia,Norway,Finland',
        min_amount=200, max_amount=500,
        handler_ids=str(razvan.id), output_team_name='CHC Bucharest',
        backup_handler_ids=str(hd['Kacper Tabiszewski'].id)))
    session.add(Rule(team_id=teams['Nordic'].id, priority=30,
        description='DK Road >500', countries='Denmark', divisions='Road', min_amount=500,
        handler_ids=f"{hd['Robert Kaczmarczyk'].id},{hd['Jakub Kowalski'].id},{hd['Karolina Rygorowicz'].id}"))
    session.add(Rule(team_id=teams['Nordic'].id, priority=50,
        description='DK Solutions', countries='Denmark', divisions='Solutions',
        handler_ids=f"{hd['Karolina Rygorowicz'].id},{hd['Jakub Kowalski'].id}"))
    session.add(Rule(team_id=teams['Nordic'].id, priority=5,
        description='DK Bestseller', countries='Denmark', customer_contains='Bestseller',
        handler_ids=str(hd['Karolina Rygorowicz'].id)))
    session.add(Rule(team_id=teams['Nordic'].id, priority=5,
        description='DK Dancover', countries='Denmark', customer_contains='Dancover',
        handler_ids=str(hd['Karolina Rygorowicz'].id)))
    session.add(Rule(team_id=teams['Nordic'].id, priority=5,
        description='DK LEGO -> Global', countries='Denmark', customer_contains='LEGO',
        output_team_name='CHC Global', output_assigned_name='#N/A'))
    session.add(Rule(team_id=teams['Nordic'].id, priority=50,
        description='Norway standard', countries='Norway',
        handler_ids=str(hd['Amelia Falk'].id)))
    session.add(Rule(team_id=teams['Nordic'].id, priority=50,
        description='EE/LT/LV standard', countries='Estonia,Lithuania,Latvia',
        handler_ids=str(hd['Weronika Kruszewska'].id),
        backup_handler_ids=str(hd['Jakub Kowalski'].id)))
    session.add(Rule(team_id=teams['Nordic'].id, priority=50,
        description='Finland Road+Solutions', countries='Finland', divisions='Road,Solutions',
        handler_ids=str(hd['Karolina Rygorowicz'].id)))
    session.add(Rule(team_id=teams['Nordic'].id, priority=40,
        description='SE Manco >=200', countries='Sweden',
        claim_sub_types='Total Missing,Partial Missing', min_amount=200,
        handler_ids=f"{hd['Michał Sztorc'].id},{hd['Tomasz Wasil'].id},{hd['Amelia Falk'].id}"))
    session.add(Rule(team_id=teams['Nordic'].id, priority=40,
        description='SE Damage >=200', countries='Sweden',
        claim_sub_types='Damage,Water Damage,Hidden Damage,Temperature Damage,Contamination', min_amount=200,
        handler_ids=f"{hd['Weronika Kruszewska'].id},{hd['Amelia Falk'].id},{hd['Tomasz Wasil'].id},{hd['Michał Sztorc'].id}"))

    # VIP
    for cust, country, hname, mn, mx, prio in [
        ('IKEA', 'Sweden', 'Weronika Kruszewska', 200, 999999, 10),
        ('Forbo', 'Sweden', 'Weronika Kruszewska', 200, 999999, 10),
        ('Ahlsell', 'Sweden', 'Weronika Kruszewska', 200, 999999, 10),
        ('Jeld-Wen', 'Sweden', 'Weronika Kruszewska', 200, 999999, 10),
        ('ICA SVERIGE', 'Sweden', 'Weronika Kruszewska', 200, 999999, 10),
        ('Spendrups', 'Sweden', 'Weronika Kruszewska', 200, 999999, 10),
        ('Coop', 'Sweden', 'Weronika Kruszewska', 0, 999999, 10),
        ('ABB ROBOTICS', None, 'Weronika Kruszewska', 0, 999999, 10),
        ('Postnord', 'Sweden', 'Amelia Falk', 200, 999999, 10),
        ('Power', 'Sweden', 'Amelia Falk', 200, 999999, 10),
        ('Bestseller', 'Denmark', 'Karolina Rygorowicz', 0, 999999, 5),
        ('Dancover', 'Denmark', 'Karolina Rygorowicz', 0, 999999, 5),
    ]:
        session.add(VIPCustomer(customer_name=cust, country=country,
            handler_id=hd[hname].id, min_amount=mn, max_amount=mx, priority=prio))

    session.commit()
    return session


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    session = get_session()

    # Admin authentication
    if 'is_admin' not in st.session_state:
        st.session_state['is_admin'] = False

    # Sidebar: Team selection + Admin login
    with st.sidebar:
        st.title("🎯 Claim Engine v2.0")
        teams = session.query(Team).all()
        team_options = {t.display_name: t.name for t in teams}
        selected_display = st.selectbox("Active Team", list(team_options.keys()),
                                        index=1 if len(team_options) > 1 else 0)
        team_name = team_options[selected_display]
        team = session.query(Team).filter_by(name=team_name).first()

        st.divider()
        page = st.radio("Navigation", [
            "📊 Process Claims",
            "📋 Rules",
            "⭐ VIP Customers",
            "🏢 Special Customers",
            "🚛 Schenker Config",
            "👥 Handlers",
            "📅 Attendance",
            "🏷️ Sub-Types",
            "📜 History",
        ])

        st.divider()

        # Admin login/logout
        admin_pw = os.environ.get("ADMIN_PASSWORD") or ""
        try:
            admin_pw = admin_pw or st.secrets.get("ADMIN_PASSWORD", "")
        except: pass

        if st.session_state['is_admin']:
            st.success("🔓 Admin")
            if st.button("Wyloguj"):
                st.session_state['is_admin'] = False
                st.rerun()
        else:
            st.caption("🔒 Tryb podglądu")
            with st.expander("🔑 Admin login"):
                pw = st.text_input("Hasło", type="password", key="admin_pw_input")
                if st.button("Zaloguj"):
                    if admin_pw and pw == admin_pw:
                        st.session_state['is_admin'] = True
                        st.rerun()
                    elif not admin_pw:
                        st.error("Brak ADMIN_PASSWORD w Secrets!")
                    else:
                        st.error("Złe hasło")

        st.divider()
        st.caption(f"v2.0 • Streamlit Edition")

    is_admin = st.session_state['is_admin']

    # ====================================================================
    # PROCESS CLAIMS
    # ====================================================================
    if page == "📊 Process Claims":
        st.header(f"📊 Process Claims — {selected_display}")

        uploaded = st.file_uploader("Upload Excel (.xlsx)", type=['xlsx'])
        if uploaded:
            df = pd.read_excel(uploaded, engine='openpyxl')
            if 'Date of Loss' in df.columns:
                df['Date of Loss'] = pd.to_datetime(df['Date of Loss'], errors='coerce', dayfirst=True)
            st.success(f"Loaded **{len(df)}** claims")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🚀 **START PROCESSING**", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    processor = ClaimProcessor(session, team)
                    result_df = processor.process_dataframe(df)
                    st.session_state['result_df'] = result_df
                    st.session_state['stats'] = processor.get_stats()

        if 'result_df' in st.session_state:
            result_df = st.session_state['result_df']
            stats = st.session_state.get('stats', {})

            st.success(f"✅ Processed **{len(result_df)}** claims")

            # Stats
            col1, col2, col3 = st.columns(3)
            assigned = len(result_df[result_df['Assigned Name'] != '#N/A']) if 'Assigned Name' in result_df.columns else 0
            unmatched = len(result_df) - assigned
            col1.metric("Total", len(result_df))
            col2.metric("Assigned", assigned)
            col3.metric("Unmatched", unmatched)

            if stats:
                st.subheader("Handler Distribution")
                stats_data = []
                for hid, cnt in sorted(stats.items(), key=lambda x: -x[1]):
                    h = session.get(Handler, hid)
                    if h: stats_data.append({"Handler": h.name, "Claims": cnt, "Team": h.team_name})
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

            # Results table
            st.subheader("Results")
            key_cols = ['Claim Import ID', 'Claim: Claim Number', 'DSV Country (Lookup)',
                        'DSV Division (Lookup)', 'Claim Sub-Type', 'Claimant Name',
                        'Claim amount EUR', 'Total liability EUR',
                        'Assigned Name', 'Claim Handler', 'Team Name', 'Assignment Reason']
            show_cols = [c for c in key_cols if c in result_df.columns]
            st.dataframe(result_df[show_cols] if show_cols else result_df,
                         use_container_width=True, height=400)

            # Downloads
            st.subheader("Download")
            c1, c2 = st.columns(2)
            csv_buf = result_df.to_csv(index=False).encode('utf-8')
            c1.download_button("📥 Download CSV", csv_buf, "claims_output.csv", "text/csv",
                               use_container_width=True)
            xlsx_buf = io.BytesIO()
            result_df.to_excel(xlsx_buf, index=False, engine='openpyxl')
            c2.download_button("📥 Download Excel", xlsx_buf.getvalue(), "claims_output.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

    # ====================================================================
    # RULES
    # ====================================================================
    elif page == "📋 Rules":
        st.header(f"📋 Rules — {selected_display}")
        st.info("Rules matched top-to-bottom by priority (lowest first). All conditions must match.")

        rules = (session.query(Rule).filter_by(team_id=team.id)
                 .order_by(Rule.priority, Rule.id).all())

        if rules:
            data = []
            for r in rules:
                data.append({
                    "ID": r.id,
                    "Prio": r.priority,
                    "Description": r.description or '',
                    "Countries": r.countries or 'ALL',
                    "Divisions": r.divisions or 'ALL',
                    "Sub-Types": r.claim_sub_types or 'ALL',
                    "Customer": r.customer_contains or '',
                    "Min EUR": f"{r.min_amount:.0f}" if r.min_amount else '',
                    "Max EUR": f"{r.max_amount:.0f}" if r.max_amount else '',
                    "Handlers": handler_names_from_ids(session, r.handler_ids),
                    "Backup": handler_names_from_ids(session, r.backup_handler_ids),
                    "Active": "✅" if r.is_active else "❌",
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=400)

        # Add / Edit / Delete
        if not is_admin:
            st.info("🔒 Zaloguj się jako admin aby edytować reguły.")
        st.subheader("Manage Rules")
        tab_add, tab_edit, tab_del = st.tabs(["➕ Add", "✏️ Edit", "🗑️ Delete"])

        all_h = all_handlers_dict(session)
        all_h_names = list(all_h.keys())
        subtypes = [st_obj.name for st_obj in session.query(ClaimSubType).filter_by(is_active=True).order_by(ClaimSubType.name).all()]

        with tab_add:
            with st.form("add_rule"):
                desc = st.text_input("Description", placeholder="e.g. Denmark Road >500 EUR")
                priority = st.number_input("Priority (lower = first)", 1, 999, 50)
                countries = st.text_input("Countries (comma-sep, empty=ALL)", placeholder="Denmark,Norway")
                col1, col2 = st.columns(2)
                divs = col1.multiselect("Divisions (empty=ALL)", DIVISIONS)
                sts = col2.multiselect("Sub-Types (empty=ALL)", subtypes)
                customer = st.text_input("Customer contains", placeholder="e.g. Bestseller")
                col1, col2 = st.columns(2)
                min_amt = col1.number_input("Min amount (>= EUR, 0=none)", 0, 9999999, 0)
                max_amt = col2.number_input("Max amount (< EUR, 0=none)", 0, 9999999, 0)
                handlers = st.multiselect("Main Handlers (load balanced)", all_h_names)
                backups = st.multiselect("Backup Handlers", all_h_names)
                col1, col2 = st.columns(2)
                out_team = col1.text_input("Output Team Name (override)", placeholder="leave empty for handler's team")
                out_assigned = col2.selectbox("Output Assigned Name", ['', '#N/A'])
                is_active = st.checkbox("Active", True)

                if st.form_submit_button("💾 Save Rule", type="primary", disabled=not is_admin):
                    r = Rule(
                        team_id=team.id, priority=priority, description=desc or None,
                        is_active=is_active,
                        countries=countries.strip() or None,
                        divisions=','.join(divs) or None,
                        claim_sub_types=','.join(sts) or None,
                        customer_contains=customer.strip() or None,
                        min_amount=min_amt if min_amt > 0 else None,
                        max_amount=max_amt if max_amt > 0 else None,
                        handler_ids=','.join(str(all_h[h]) for h in handlers) or None,
                        backup_handler_ids=','.join(str(all_h[h]) for h in backups) or None,
                        output_team_name=out_team.strip() or None,
                        output_assigned_name=out_assigned or None,
                    )
                    session.add(r); session.commit()
                    st.success("Rule added!"); st.rerun()

        with tab_edit:
            if rules:
                rule_options = {f"#{r.id} [P{r.priority}] {r.description or r.countries or '?'}": r.id for r in rules}
                sel = st.selectbox("Select rule to edit", list(rule_options.keys()), key="edit_rule_sel")
                rid = rule_options[sel]
                rule = session.get(Rule, rid)

                with st.form("edit_rule"):
                    desc = st.text_input("Description", rule.description or '')
                    priority = st.number_input("Priority", 1, 999, rule.priority)
                    countries = st.text_input("Countries", rule.countries or '')
                    col1, col2 = st.columns(2)
                    cur_divs = [d.strip() for d in (rule.divisions or '').split(',') if d.strip()]
                    divs = col1.multiselect("Divisions", DIVISIONS, default=cur_divs, key="ed_div")
                    cur_sts = [s.strip() for s in (rule.claim_sub_types or '').split(',') if s.strip()]
                    sts = col2.multiselect("Sub-Types", subtypes, default=[s for s in cur_sts if s in subtypes], key="ed_st")
                    customer = st.text_input("Customer contains", rule.customer_contains or '')
                    col1, col2 = st.columns(2)
                    min_amt = col1.number_input("Min EUR", 0, 9999999, int(rule.min_amount or 0), key="ed_mn")
                    max_amt = col2.number_input("Max EUR", 0, 9999999, int(rule.max_amount or 0), key="ed_mx")

                    cur_main = []
                    if rule.handler_ids:
                        for hid in rule.handler_ids.split(','):
                            if hid.strip().isdigit():
                                h = session.get(Handler, int(hid))
                                if h:
                                    key = f"{h.name} [{h.team_name}]"
                                    if key in all_h_names: cur_main.append(key)
                    handlers = st.multiselect("Main Handlers", all_h_names, default=cur_main, key="ed_hnd")

                    cur_bk = []
                    if rule.backup_handler_ids:
                        for hid in rule.backup_handler_ids.split(','):
                            if hid.strip().isdigit():
                                h = session.get(Handler, int(hid))
                                if h:
                                    key = f"{h.name} [{h.team_name}]"
                                    if key in all_h_names: cur_bk.append(key)
                    backups = st.multiselect("Backup", all_h_names, default=cur_bk, key="ed_bk")

                    col1, col2 = st.columns(2)
                    out_team = col1.text_input("Output Team", rule.output_team_name or '', key="ed_ot")
                    oa_idx = ['', '#N/A'].index(rule.output_assigned_name) if rule.output_assigned_name in ['', '#N/A'] else 0
                    out_assigned = col2.selectbox("Output Assigned", ['', '#N/A'], index=oa_idx, key="ed_oa")
                    is_active = st.checkbox("Active", rule.is_active, key="ed_act")

                    if st.form_submit_button("💾 Update Rule", type="primary", disabled=not is_admin):
                        rule.description = desc or None; rule.priority = priority
                        rule.countries = countries.strip() or None
                        rule.divisions = ','.join(divs) or None
                        rule.claim_sub_types = ','.join(sts) or None
                        rule.customer_contains = customer.strip() or None
                        rule.min_amount = min_amt if min_amt > 0 else None
                        rule.max_amount = max_amt if max_amt > 0 else None
                        rule.handler_ids = ','.join(str(all_h[h]) for h in handlers) or None
                        rule.backup_handler_ids = ','.join(str(all_h[h]) for h in backups) or None
                        rule.output_team_name = out_team.strip() or None
                        rule.output_assigned_name = out_assigned or None
                        rule.is_active = is_active
                        session.commit(); st.success("Updated!"); st.rerun()

        with tab_del:
            if rules:
                rule_del_opts = {f"#{r.id} [P{r.priority}] {r.description or r.countries or '?'}": r.id for r in rules}
                sel = st.selectbox("Select rule to delete", list(rule_del_opts.keys()), key="del_rule_sel")
                if st.button("🗑️ Delete Rule", disabled=not is_admin, type="secondary"):
                    r = session.get(Rule, rule_del_opts[sel])
                    if r: session.delete(r); session.commit(); st.success("Deleted!"); st.rerun()

    # ====================================================================
    # VIP CUSTOMERS
    # ====================================================================
    elif page == "⭐ VIP Customers":
        st.header("⭐ VIP Customers")
        if not is_admin: st.info("🔒 Zaloguj się jako admin aby edytować.")
        st.info("Checked BEFORE rules. Matched by claimant name + country + amount range.")

        vips = session.query(VIPCustomer).order_by(VIPCustomer.priority).all()
        if vips:
            data = [{"ID": v.id, "Prio": v.priority, "Customer": v.customer_name,
                     "Country": v.country or 'ANY',
                     "Handler": (session.get(Handler, v.handler_id).name if session.get(Handler, v.handler_id) else '?'),
                     "Min EUR": f"{v.min_amount:.0f}", "Max EUR": f"{v.max_amount:.0f}",
                     "Active": "✅" if v.is_active else "❌"} for v in vips]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        all_h = all_handlers_dict(session)
        tab_add, tab_edit, tab_del = st.tabs(["➕ Add", "✏️ Edit", "🗑️ Delete"])

        with tab_add:
            with st.form("add_vip"):
                name = st.text_input("Customer name (contains)")
                country = st.text_input("Country (empty=any)")
                handler = st.selectbox("Handler", list(all_h.keys()))
                col1, col2, col3 = st.columns(3)
                mn = col1.number_input("Min EUR", 0, 999999, 0)
                mx = col2.number_input("Max EUR", 0, 999999, 999999)
                prio = col3.number_input("Priority", 1, 999, 10)
                if st.form_submit_button("💾 Save", disabled=not is_admin):
                    session.add(VIPCustomer(customer_name=name, country=country.strip() or None,
                        handler_id=all_h[handler], min_amount=mn, max_amount=mx, priority=prio))
                    session.commit(); st.success("Added!"); st.rerun()

        with tab_edit:
            if vips:
                opts = {f"{v.customer_name} ({v.country or 'ANY'})": v.id for v in vips}
                sel = st.selectbox("Select VIP", list(opts.keys()), key="edit_vip")
                vip = session.get(VIPCustomer, opts[sel])
                with st.form("edit_vip"):
                    name = st.text_input("Customer", vip.customer_name)
                    country = st.text_input("Country", vip.country or '')
                    cur_h = session.get(Handler, vip.handler_id)
                    cur_key = f"{cur_h.name} [{cur_h.team_name}]" if cur_h else list(all_h.keys())[0]
                    handler = st.selectbox("Handler", list(all_h.keys()),
                        index=list(all_h.keys()).index(cur_key) if cur_key in all_h else 0, key="ev_h")
                    col1, col2, col3 = st.columns(3)
                    mn = col1.number_input("Min", 0, 999999, int(vip.min_amount), key="ev_mn")
                    mx = col2.number_input("Max", 0, 999999, int(vip.max_amount), key="ev_mx")
                    prio = col3.number_input("Prio", 1, 999, vip.priority, key="ev_p")
                    if st.form_submit_button("💾 Update", disabled=not is_admin):
                        vip.customer_name = name; vip.country = country.strip() or None
                        vip.handler_id = all_h[handler]; vip.min_amount = mn
                        vip.max_amount = mx; vip.priority = prio
                        session.commit(); st.success("Updated!"); st.rerun()

        with tab_del:
            if vips:
                opts = {f"{v.customer_name} ({v.country or 'ANY'})": v.id for v in vips}
                sel = st.selectbox("Select VIP to delete", list(opts.keys()), key="del_vip")
                if st.button("🗑️ Delete VIP", disabled=not is_admin):
                    v = session.get(VIPCustomer, opts[sel])
                    if v: session.delete(v); session.commit(); st.rerun()

    # ====================================================================
    # SPECIAL CUSTOMERS
    # ====================================================================
    elif page == "🏢 Special Customers":
        st.header("🏢 Special Customers (Global)")
        if not is_admin: st.info("🔒 Zaloguj się jako admin aby edytować.")
        st.info("Checked FIRST. e.g. Abbott, LEGO, Adidas → dedicated handlers.")

        specials = session.query(SpecialCustomer).order_by(SpecialCustomer.customer_name).all()
        if specials:
            data = [{"ID": s.id, "Customer": s.customer_name,
                     "Handlers": handler_names_from_ids(session, s.handler_ids),
                     "Active": "✅" if s.is_active else "❌"} for s in specials]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        all_h = all_handlers_dict(session)
        tab_add, tab_del = st.tabs(["➕ Add", "🗑️ Delete"])

        with tab_add:
            with st.form("add_special"):
                name = st.text_input("Customer name (contains)")
                handlers = st.multiselect("Handlers", list(all_h.keys()))
                if st.form_submit_button("💾 Save", disabled=not is_admin):
                    hids = ','.join(str(all_h[h]) for h in handlers)
                    session.add(SpecialCustomer(customer_name=name, handler_ids=hids))
                    session.commit(); st.success("Added!"); st.rerun()

        with tab_del:
            if specials:
                opts = {s.customer_name: s.id for s in specials}
                sel = st.selectbox("Select to delete", list(opts.keys()), key="del_sc")
                if st.button("🗑️ Delete", disabled=not is_admin):
                    s = session.get(SpecialCustomer, opts[sel])
                    if s: session.delete(s); session.commit(); st.rerun()

    # ====================================================================
    # SCHENKER CONFIG
    # ====================================================================
    elif page == "🚛 Schenker Config":
        st.header("🚛 Schenker Merge Configuration")
        if not is_admin: st.info("🔒 Zaloguj się jako admin aby edytować.")
        st.info("Shipments without DSV dash are checked here. Country + division + merge month in 2025.")

        configs = session.query(SchenkerConfig).filter_by(is_active=True).order_by(
            SchenkerConfig.merge_month, SchenkerConfig.country).all()
        if configs:
            data = [{"ID": c.id, "Country": c.country, "Division": c.division,
                     "Merge Month (2025)": c.merge_month} for c in configs]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        tab_add, tab_del = st.tabs(["➕ Add", "🗑️ Delete"])
        with tab_add:
            with st.form("add_sch"):
                country = st.text_input("Country")
                div = st.selectbox("Division", ['all'] + DIVISIONS)
                month = st.number_input("Merge Month (2025)", 1, 12, 8)
                if st.form_submit_button("💾 Save", disabled=not is_admin):
                    session.add(SchenkerConfig(country=country, division=div, merge_month=month))
                    session.commit(); st.success("Added!"); st.rerun()
        with tab_del:
            if configs:
                opts = {f"{c.country} / {c.division} / month {c.merge_month}": c.id for c in configs}
                sel = st.selectbox("Select", list(opts.keys()), key="del_sch")
                if st.button("🗑️ Delete", disabled=not is_admin):
                    c = session.get(SchenkerConfig, opts[sel])
                    if c: session.delete(c); session.commit(); st.rerun()

    # ====================================================================
    # HANDLERS
    # ====================================================================
    elif page == "👥 Handlers":
        st.header(f"👥 Handlers — {selected_display}")
        if not is_admin: st.info("🔒 Zaloguj się jako admin aby edytować.")

        handlers = session.query(Handler).filter_by(team_id=team.id).order_by(Handler.team_name, Handler.name).all()
        if handlers:
            data = [{"Name": h.name, "Riskonnect ID": h.riskonnect_id,
                     "Team": h.team_name or '',
                     "Backup": h.backup_handler.name if h.backup_handler else '-',
                     "Status": "✅ Present" if h.is_present else "❌ Absent"} for h in handlers]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        tab_add, tab_edit, tab_del = st.tabs(["➕ Add", "✏️ Edit", "🗑️ Delete"])

        all_handlers_for_backup = {f"{h.name} [{h.team_name}]": h.id
            for h in session.query(Handler).order_by(Handler.name).all()}

        with tab_add:
            with st.form("add_handler"):
                name = st.text_input("Name")
                rid = st.text_input("Riskonnect ID")
                tname = st.selectbox("Team Name", TEAM_NAMES[:4])
                backup = st.selectbox("Backup Handler", ['-- None --'] + list(all_handlers_for_backup.keys()))
                if st.form_submit_button("💾 Save", disabled=not is_admin):
                    if name and rid:
                        bk_id = all_handlers_for_backup.get(backup) if backup != '-- None --' else None
                        session.add(Handler(name=name, riskonnect_id=rid, team_name=tname,
                            team_id=team.id, backup_handler_id=bk_id))
                        session.commit(); st.success(f"Added {name}!"); st.rerun()

        with tab_edit:
            if handlers:
                opts = {f"{h.name} ({h.riskonnect_id})": h.id for h in handlers}
                sel = st.selectbox("Select handler", list(opts.keys()), key="edit_h")
                h = session.get(Handler, opts[sel])
                with st.form("edit_handler"):
                    name = st.text_input("Name", h.name)
                    rid = st.text_input("Riskonnect ID", h.riskonnect_id)
                    tname = st.selectbox("Team", TEAM_NAMES[:4],
                        index=TEAM_NAMES[:4].index(h.team_name) if h.team_name in TEAM_NAMES[:4] else 0, key="eh_t")
                    bk_opts = ['-- None --'] + list(all_handlers_for_backup.keys())
                    cur_bk = '-- None --'
                    if h.backup_handler:
                        key = f"{h.backup_handler.name} [{h.backup_handler.team_name}]"
                        if key in bk_opts: cur_bk = key
                    backup = st.selectbox("Backup", bk_opts, index=bk_opts.index(cur_bk), key="eh_bk")
                    if st.form_submit_button("💾 Update", disabled=not is_admin):
                        h.name = name; h.riskonnect_id = rid; h.team_name = tname
                        h.backup_handler_id = all_handlers_for_backup.get(backup) if backup != '-- None --' else None
                        session.commit(); st.success("Updated!"); st.rerun()

        with tab_del:
            if handlers:
                opts = {f"{h.name} ({h.riskonnect_id})": h.id for h in handlers}
                sel = st.selectbox("Select to delete", list(opts.keys()), key="del_h")
                if st.button("🗑️ Delete Handler", disabled=not is_admin):
                    h = session.get(Handler, opts[sel])
                    if h: session.delete(h); session.commit(); st.rerun()

    # ====================================================================
    # ATTENDANCE
    # ====================================================================
    elif page == "📅 Attendance":
        st.header(f"📅 Attendance — {selected_display}")
        if not is_admin: st.info("🔒 Zaloguj się jako admin aby edytować.")

        handlers = session.query(Handler).filter_by(team_id=team.id).order_by(Handler.team_name, Handler.name).all()

        col1, col2, _ = st.columns([1, 1, 3])
        if col1.button("✅ All Present", disabled=not is_admin):
            for h in handlers: h.is_present = True
            session.commit(); st.rerun()
        if col2.button("❌ All Absent", disabled=not is_admin):
            for h in handlers: h.is_present = False
            session.commit(); st.rerun()

        current_team = None
        for h in handlers:
            if h.team_name != current_team:
                current_team = h.team_name
                st.subheader(current_team or 'Unknown')

            new_val = st.checkbox(
                f"{h.name}  `{h.riskonnect_id}`",
                value=h.is_present,
                key=f"att_{h.id}",
                disabled=not is_admin
            )
            if new_val != h.is_present:
                h.is_present = new_val
                session.commit()

    # ====================================================================
    # SUB-TYPES
    # ====================================================================
    elif page == "🏷️ Sub-Types":
        st.header("🏷️ Claim Sub-Types")
        if not is_admin: st.info("🔒 Zaloguj się jako admin aby edytować.")

        subtypes = session.query(ClaimSubType).order_by(ClaimSubType.category, ClaimSubType.name).all()
        if subtypes:
            data = [{"Name": s.name, "Category": s.category or '', "Active": "✅" if s.is_active else "❌"}
                    for s in subtypes]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.form("add_st"):
                name = st.text_input("Name")
                cat = st.selectbox("Category", ['damage', 'manco', 'other', ''])
                if st.form_submit_button("➕ Add", disabled=not is_admin):
                    session.add(ClaimSubType(name=name, category=cat or None))
                    session.commit(); st.rerun()
        with col2:
            if subtypes:
                opts = {s.name: s.id for s in subtypes}
                sel = st.selectbox("Delete sub-type", list(opts.keys()))
                if st.button("🗑️ Delete", disabled=not is_admin):
                    s = session.get(ClaimSubType, opts[sel])
                    if s: session.delete(s); session.commit(); st.rerun()

    # ====================================================================
    # HISTORY
    # ====================================================================
    elif page == "📜 History":
        st.header("📜 Processing History")

        history = session.query(History).order_by(History.timestamp.desc()).limit(300).all()
        if history:
            data = [{"Time": h.timestamp.strftime('%Y-%m-%d %H:%M') if h.timestamp else '',
                     "Claim #": h.claim_number, "Country": h.country,
                     "Division": h.division, "Claimant": (h.claimant or '')[:30],
                     "Handler": h.handler_name, "Team": h.team_name,
                     "Reason": h.reason} for h in history]
            st.dataframe(pd.DataFrame(data), use_container_width=True, height=500, hide_index=True)
        else:
            st.info("No history yet. Process some claims first.")

        if st.button("🗑️ Clear All History", disabled=not is_admin):
            session.query(History).delete(); session.commit(); st.rerun()


if __name__ == '__main__':
    main()
