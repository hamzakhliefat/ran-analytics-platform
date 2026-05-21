# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:15:32 2026

@author: hamza khlefat
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class RawCols:
    cell_id: str = "CEll ID"
    date: str = "Date Id"
    gov: str = "Gov"
    site_id: str = "Site ID"

    throughput: str = "NPM DL User Throughput"
    rrc_users_max: str = "NPM RRC Users Max"
    prb_dl: str = "NPM DL PRB Utilization"
    drop_rate: str = "NPM Service Drop Rate"

    ho_exe_succ: str = "Pm Ho Exe Succ"
    ho_prep_att: str = "Pm Ho Prep Att"

    traffic_dl_gb: str = "NPM DL Traffic GB"
    harq_ack_64qam: str = "pmMacHarqDlAck64qam"


@dataclass(frozen=True)
class Cols:
    cell_id: str = "cell_id"
    date: str = "date"
    gov: str = "gov"
    site_id: str = "site_id"

    throughput: str = "NPM_DL_User_Throughput"
    rrc_users_max: str = "NPM_RRC_Users_Max"
    prb_dl: str = "NPM_DL_PRB_Utilization"
    drop_rate: str = "NPM_Service_Drop_Rate"

    ho_exe_succ: str = "pmHoExeSucc"
    ho_prep_att: str = "pmHoPrepAtt"
    ho_sr: str = "HO_SR"

    traffic_dl_gb: str = "NPM_DL_Traffic_GB"
    harq_ack_64qam: str = "pmMacHarqDlAck64qam"




def rename_map(raw: RawCols, std: Cols) -> Dict[str, str]:
    return {
        raw.cell_id: std.cell_id,
        raw.date: std.date,
        raw.gov: std.gov,
        raw.site_id: std.site_id,

        raw.throughput: std.throughput,
        raw.rrc_users_max: std.rrc_users_max,
        raw.prb_dl: std.prb_dl,
        raw.drop_rate: std.drop_rate,

        raw.ho_exe_succ: std.ho_exe_succ,
        raw.ho_prep_att: std.ho_prep_att,

        raw.traffic_dl_gb: std.traffic_dl_gb,
        raw.harq_ack_64qam: std.harq_ack_64qam,
    }


@dataclass(frozen=True)
class Thresholds:
    ho_sr_low: float = 0.95
    prb_high: float = 0.85
    drop_anomaly_quantile: float = 0.95


@dataclass(frozen=True)
class DB:
    path: str = "db/ran_kpis.sqlite"
    table: str = "kpi_clean"


def numeric_cols(std: Cols) -> List[str]:
    return [
        std.throughput,
        std.rrc_users_max,
        std.prb_dl,
        std.drop_rate,
        std.ho_exe_succ,
        std.ho_prep_att,
        std.traffic_dl_gb,
        std.harq_ack_64qam,
    ]
