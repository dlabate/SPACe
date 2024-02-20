import argparse
from typing import Union, Optional, List, Tuple
from pathlib import PosixPath, WindowsPath, Path

import numpy as np
import pandas as pd

import tifffile
import multiprocessing as mp


parser = argparse.ArgumentParser(description='Cell-paint')
args = parser.parse_args()


if args.experiment == "2021-CP007" or args.experiment == "2021-CP008":
    args.channel_dies = {
        "C1": "DAPI",
        "C2": "ConA+Syto14",
        "C3": "WGA+Phalloidin",
        "C4": "MitoTracker"}
elif args.experiment == "others":
    "paper title: Morphological profiling of environmental chemicals enables efficient and"
    "untargeted exploration of combination effects"
    "paper journal: Science of the Total Environment"
    "Department of Pharmaceutical Biosciences and Science for Life Laboratory, Uppsala University, Sweden"
    # Fluorescence microscopy was conducted using a high throughput ImageXpressMicro XLS(Molecular Devices)
    # microscope with a 20×objective with laser - based autofocus.
    args.nucleus_idx = 0
    args.cyto_idx = 4
    args.nucleoli_idx = 3
    args.actin_idx = 2
    args.mito_idx = 1
    args.channel_dies = {
        "C1": "Hoechst",
        "C2": "MitoTracker",
        "C3": "WGA+Phalloidin",
        "C4": "Syto14",
        "C5": "Concanavalin A", }
    args.cellpose_nucleus_diam = 80
    args.nucleoli_channel_rescale_intensity_percentile_ub = 99.8
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .21
    args.min_nucleoli_size_multiplier = .001
    args.max_nucleoli_size_multiplier = .5
else:
    args.channel_dies = {
        "C1": "DAPI",
        "C2": "Concanavalin A",
        "C3": "Syto14",
        "C4": "WGA+Phalloidin",
        "C5": "MitoTracker"}
    # if args.experiment == "Bolt-MCF7-CellPaint_20220929_160725":
    #     args.expid = "2022-09-29-Bolt-MCF7-CellPaint"
    # elif args.experiment == "Bolt-Seema-CellPaint_20220930_102103":
    #     args.expid = "2022-09-30-CP-Bolt-Seema"
    # elif args.experiment == "20220920-Bolt-seema_20220920_164521":
    #     args.expid = "2022-09-20-CP-Bolt-Seema"
    # elif args.experiment == "20220912-CP-Bolt_20220912_153142":
    #     args.expid = "2022-09-12-CP-Bolt"
    # elif args.experiment == "20220908-CP-benchmark-DRC-replicate2_20220908_142836":
    #     args.expid = "20220909-CP-231-PANC1"
    # elif args.experiment == "20220908-CP-benchmark-DRC-replicate2_20220908_142836":
    #     args.expid = "2022-09-08-CP-BM-DRC2"
    # elif args.experiment == "20220831-cellpainting-benchmarking-DRC_20220831_173200":
    #     args.expid = "2022-08-31-CP-BM-DRC"
    # elif args.experiment == "2022-0817-CP-benchmarking-density_20220817_120119":
    #     args.expid = "2022-08-17-CP-BM-density"
    # elif args.experiment == "2022-0810-CP-benchmarking-WGArestain_20220810_121628":
    #     args.expid = "2022-08-10-CP-BM-WGA-restain"

if args.experiment == "20220607-U2OS-density-5channel_20220607_135315":
    def get_wellid2treatment():
        platemap = pd.read_excel(
            args.main_path / args.experiment / "U2OS_Cell_Density-CellPaint-5channel-imaging.xlsx",
            sheet_name="Sheet1")
        wellid2treat = {}
        for ii in range(2, 12):
            row_id = platemap.iloc[ii]["Cell Density:"]
            for jj, item in enumerate(platemap.iloc[ii][1:]):
                if str(item) != "nan":
                    wellid2treat[f"{row_id}{str(jj + 1).zfill(2)}"] = str(item).rstrip(".0")
        return wellid2treat
    args.lab = "baylor"
    args.expid = "2022-06-07-U2OS-Density-BM"
    args.img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
    args.plate_protocol = "PerkinElmer"

    args.wellid2treatment = get_wellid2treatment()
    args.wellids = list(np.unique(list(args.wellid2treatment.keys())))
    args.treatments = list(np.unique(list(args.wellid2treatment.values())))
    args.celllines = ["U2OS"]
    args.wellid2cellline = {key: "U2OS" for key in args.wellid2treatment.keys()}
    args.wellid2dosage = {key: "N/A" for key in args.wellid2treatment.keys()}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))

elif args.experiment == "20220603-Shelton-PC3-CP_20220603_151615":
    def get_wellid2treatment():
        platemap = pd.read_excel(
            args.main_path / args.experiment / "Shelton_platemaps.xlsx", sheet_name="060122")
        wellid2treat = {}
        for ii in range(2, len(platemap) - 4):
            row_id = platemap.iloc[ii]["Shelton Boyd/Damien Young"]
            for jj, item in enumerate(platemap.iloc[ii][2:]):
                if str(item) != "nan":
                    wellid2treat[f"{row_id}{str(jj + 1).zfill(2)}"] = str(item).lstrip().rstrip()
        return wellid2treat

    args.lab = "baylor"
    args.expid = "2022-06-03-Shelton-PC3"
    args.img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
    args.plate_protocol = "PerkinElmer"

    args.wellid2treatment = get_wellid2treatment()
    args.wellids = list(np.unique(list(args.wellid2treatment.keys())))
    args.treatments = list(np.unique(list(args.wellid2treatment.values())))
    args.celllines = ["PC3"]
    args.wellid2cellline = {key: "PC3" for key in args.wellid2treatment.keys()}
    args.wellid2dosage = {key: "N/A" for key in args.wellid2treatment.keys()}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))

elif args.experiment == "20220617-Shelton-exp2_20220617_110944":
    def get_wellid2treatment():
        platemap = pd.read_excel(
            args.main_path / args.experiment / "Shelton_platemaps.xlsx", sheet_name="061422")
        wellid2treat = {}
        for ii in range(2, len(platemap) - 4):
            row_id = platemap.iloc[ii]["Shelton Boyd/Damien Young"]
            for jj, item in enumerate(platemap.iloc[ii][2:]):
                if str(item) != "nan":
                    wellid2treat[f"{row_id}{str(jj + 1).zfill(2)}"] = str(item).lstrip().rstrip()
        return wellid2treat

    args.lab = "baylor"
    args.expid = "2022-06-17-Shelton-PC3-2"
    args.img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
    args.plate_protocol = "PerkinElmer"

    args.wellid2treatment = get_wellid2treatment()
    args.wellids = list(np.unique(list(args.wellid2treatment.keys())))
    args.treatments = list(np.unique(list(args.wellid2treatment.values())))
    args.celllines = ["PC3"]
    args.wellid2cellline = {key: "PC3" for key in args.wellid2treatment.keys()}
    args.wellid2dosage = {
        "A01": "10uM", "B01": "10uM", "C01": "10uM", "D01": "10uM", "E01": "10uM", "F01": "10uM",
        "A02": "1uM", "B02": "1uM", "C02": "1uM", "D02": "1uM", "E02": "1uM", "F02": "1uM",
        "A03": "100nM", "B03": "100nM", "C03": "100nM", "D03": "100nM", "E03": "100nM", "F03": "100nM",

        "A04": "10uM", "B04": "10uM", "C04": "10uM", "D04": "10uM", "E04": "10uM", "F04": "10uM",
        "A05": "1uM", "B05": "1uM", "C05": "1uM", "D05": "1uM", "E05": "1uM", "F05": "1uM",
        "A06": "100nM", "B06": "100nM", "C06": "100nM", "D06": "100nM", "E06": "100nM", "F06": "100nM",

        "A07": "10uM", "B07": "10uM", "C07": "10uM", "D07": "10uM", "E07": "10uM", "F07": "10uM",
        "A08": "1uM", "B08": "1uM", "C08": "1uM", "D08": "1uM", "E08": "1uM", "F08": "1uM",
        "A09": "100nM", "B09": "100nM", "C09": "100nM", "D09": "100nM", "E09": "100nM", "F09": "100nM",

        "A10": "10uM", "B10": "10uM", "C10": "10uM", "D10": "10uM", "E10": "10uM", "F10": "10uM",
        "A11": "1uM", "B11": "1uM", "C11": "1uM", "D11": "1uM", "E11": "1uM", "F11": "1uM",
        "A12": "100nM", "B12": "100nM", "C12": "100nM", "D12": "100nM", "E12": "100nM", "F12": "100nM",
        "G01": "N/A"}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))


elif args.experiment == "20220714-Shelton-Exp03_20220714_132321":
    def get_wellid2treatment():
        platemap = pd.read_excel(
            args.main_path / args.experiment / "Shelton_platemaps.xlsx", sheet_name="061422")
        wellid2treat = {}
        for ii in range(2, len(platemap) - 4):
            row_id = platemap.iloc[ii]["Shelton Boyd/Damien Young"]
            for jj, item in enumerate(platemap.iloc[ii][2:]):
                if str(item) != "nan":
                    wellid2treat[f"{row_id}{str(jj + 1).zfill(2)}"] = str(item).lstrip().rstrip()
        return wellid2treat

    args.lab = "baylor"
    args.expid = "2022-07-14-Shelton-PC3-3"
    args.img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
    args.plate_protocol = "PerkinElmer"

    args.cellpose_nucleus_diam = 80
    args.wellid2treatment = get_wellid2treatment()
    args.wellids = list(np.unique(list(args.wellid2treatment.keys())))
    args.treatments = list(np.unique(list(args.wellid2treatment.values())))
    args.celllines = ["PC3"]
    args.wellid2cellline = {key: "PC3" for key in args.wellid2treatment.keys()}
    args.wellid2dosage = {
        "A01": "10uM", "B01": "10uM", "C01": "10uM", "D01": "10uM", "E01": "10uM", "F01": "10uM",
        "A02": "1uM", "B02": "1uM", "C02": "1uM", "D02": "1uM", "E02": "1uM", "F02": "1uM",
        "A03": "100nM", "B03": "100nM", "C03": "100nM", "D03": "100nM", "E03": "100nM", "F03": "100nM",

        "A04": "10uM", "B04": "10uM", "C04": "10uM", "D04": "10uM", "E04": "10uM", "F04": "10uM",
        "A05": "1uM", "B05": "1uM", "C05": "1uM", "D05": "1uM", "E05": "1uM", "F05": "1uM",
        "A06": "100nM", "B06": "100nM", "C06": "100nM", "D06": "100nM", "E06": "100nM", "F06": "100nM",

        "A07": "10uM", "B07": "10uM", "C07": "10uM", "D07": "10uM", "E07": "10uM", "F07": "10uM",
        "A08": "1uM", "B08": "1uM", "C08": "1uM", "D08": "1uM", "E08": "1uM", "F08": "1uM",
        "A09": "100nM", "B09": "100nM", "C09": "100nM", "D09": "100nM", "E09": "100nM", "F09": "100nM",

        "A10": "10uM", "B10": "10uM", "C10": "10uM", "D10": "10uM", "E10": "10uM", "F10": "10uM",
        "A11": "1uM", "B11": "1uM", "C11": "1uM", "D11": "1uM", "E11": "1uM", "F11": "1uM",
        "A12": "100nM", "B12": "100nM", "C12": "100nM", "D12": "100nM", "E12": "100nM", "F12": "100nM",
        "G01": "N/A"}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))
elif args.experiment == "others":
    platemap = pd.read_csv(args.main_path / args.experiment / "platemap.csv")
    # for now we work only with U2OS
    platemap = platemap.loc[platemap["cellLine"] == "U2OS"]

    args.lab = "others"
    args.expid = "P000025-combchem-v3-U2OS-24h-L1-copy1"
    args.img_folder = "P000025-combchem-v3-U2OS-24h-L1-copy1"
    args.plate_protocol = "combchem"

    args.celllines = ["U2OS", ]
    args.wellids = np.unique(platemap["well_id"])

    platemap.loc[pd.isnull(platemap["treat_id"]), "treat_id"] = ""
    platemap.loc[platemap["pert_type"] == "blank", "pert_type"] = "media"

    platemap.loc[pd.isnull(platemap["cmpd_conc_uM"]), "cmpd_conc_uM"] = 0
    platemap["cmpd_conc_uM"] = platemap["cmpd_conc_uM"].astype(str)
    platemap.loc[pd.isnull(platemap["conc_units"]), "conc_units"] = "uM"

    platemap["all_treatments"] = platemap["pert_type"] + platemap["treat_id"]
    platemap["all_dosages"] = platemap["cmpd_conc_uM"] + platemap["conc_units"]
    args.treatments = np.unique(platemap["all_treatments"])
    args.treatments = ['media'] + list(np.unique([it.split("_")[0] for it in args.treatments[1:]]))
    args.dosages = np.unique(platemap["all_dosages"])
    args.densities = [0]

    args.wellid2treatment = {
        it: platemap.loc[platemap["well_id"] == it, "all_treatments"].values[0].split("_")[0]
        for it in platemap["well_id"]}
    args.wellid2dosage = {
        it: platemap.loc[platemap["well_id"] == it, "all_dosages"].values[0].split("_")[0]
        for it in platemap["well_id"]}
    args.wellid2cellline = {
        it: platemap.loc[platemap["well_id"] == it, "cellLine"].values[0].split("_")[0]
        for it in platemap["well_id"]}
    args.wellid2density = {it: 0 for it in platemap['well_id']}
    # for key, val in args.wellid2cellline.items():
    #     print(key, val)
    # print(args.wellid2treatment)

elif args.experiment == "2021-CP007":
    def get_wellid2treatment():
        platemap = pd.read_excel(args.main_path / args.experiment / "Compound_whole_Plate_Layout.xlsx",
                                 sheet_name="Sheet1")
        wellid2treat = {}
        for ii in range(2, len(platemap) - 2):
            row_id = platemap.iloc[ii]["Compound:"]
            for jj, item in enumerate(platemap.iloc[ii][1:]):
                wellid2treat[f"{row_id}{str(jj + 1).zfill(2)}"] = item.lstrip().rstrip()
        return wellid2treat

    args.lab = "baylor"
    args.expid = "2021-CP007"
    args.img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
    args.plate_protocol = "PerkinElmer"

    args.wellid2treatment = get_wellid2treatment()
    args.wellids = list(np.unique(list(args.wellid2treatment.keys())))
    args.treatments = np.unique(list(args.wellid2treatment.values()))
    # args.treatments = sorted([item[0:SHORT_INDEX] for item in np.unique(list(args.wellid2treatment.values()))])
    args.celllines = ["A549", "HELA", "JEG3", "SKBR-3"]
    CP007 = {"A": "SKBR-3", "B": "SKBR-3", "C": "SKBR-3", "D": "SKBR-3",
             "E": "A549", "F": "A549", "G": "A549", "H": "A549",
             "I": "HELA", "J": "HELA", "K": "HELA", "L": "HELA",
             "M": "JEG3", "N": "JEG3", "O": "JEG3", "P": "JEG3"}
    args.wellid2cellline = {key: CP007[key[0]] for key in args.wellid2treatment.keys()}
    args.wellid2dosage = {key: "N/A" for key in args.wellid2treatment.keys()}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))
elif args.experiment == "2021-CP008":
    def get_wellid2treatment():
        platemap = pd.read_excel(args.main_path / args.experiment / "Compound_whole_Plate_Layout.xlsx",
                                 sheet_name="Sheet1")
        wellid2treat = {}
        for ii in range(2, len(platemap) - 2):
            row_id = platemap.iloc[ii]["Compound:"]
            for jj, item in enumerate(platemap.iloc[ii][1:]):
                wellid2treat[f"{row_id}{str(jj + 1).zfill(2)}"] = item.lstrip().rstrip()
        return wellid2treat

    args.lab = "baylor"
    args.expid = "2021-CP008"
    args.img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
    args.plate_protocol = "PerkinElmer"

    args.wellid2treatment = get_wellid2treatment()
    args.wellids = list(np.unique(list(args.wellid2treatment.keys())))
    args.treatments = np.unique(list(args.wellid2treatment.values()))
    # args.treatments = sorted([item[0:SHORT_INDEX] for item in np.unique(list(args.wellid2treatment.values()))])
    args.celllines = ["HepG2", "PC3", "RT4", "U2OS"]
    CP008 = {"A": "HepG2", "B": "HepG2", "C": "HepG2", "D": "HepG2",
             "E": "PC3", "F": "PC3", "G": "PC3", "H": "PC3",
             "I": "RT4", "J": "RT4", "K": "RT4", "L": "RT4",
             "M": "U2OS", "N": "U2OS", "O": "U2OS", "P": "U2OS"}
    args.wellid2cellline = {key: CP008[key[0]] for key in args.wellid2treatment.keys()}
    args.wellid2dosage = {key: "N/A" for key in args.wellid2treatment.keys()}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))
elif args.experiment == "20220623-CAMII-Seema-cellpainting_20220623_101847":
    def get_wellid2treatment():
        platemap = pd.read_excel(
            args.main_path / args.experiment / "220621_HT29_Cellpaint_platelayout.xlsx", sheet_name="Sheet1")
        wellid2treat = {}
        wellid2celline = {}
        for it1, it2, it3, it4 in zip(platemap['Row'], platemap['Col'], platemap['Cell Line'], platemap['Treatment']):
            wellid2treat[f"{it1}{str(it2).zfill(2)}"] = str(it4).lstrip().rstrip()
            wellid2celline[f"{it1}{str(it2).zfill(2)}"] = str(it3).lstrip().rstrip()

        cellines = np.unique(list(wellid2celline.values()))
        treatments = np.unique(list(wellid2treat.values()))
        wellids = np.unique(list(wellid2treat.keys()))
        return wellid2treat, wellids, treatments, cellines, wellid2celline

    args.lab = "ibt"
    args.expid = "20220623-CAMII-Seema"
    args.img_folder = "AssayPlate_Greiner_#781091"
    args.plate_protocol = "Greiner"

    args.wellid2treatment, args.wellids, args.treatments, args.celllines, args.wellid2cellline = get_wellid2treatment()
    args.wellid2dosage = {key: "N/A" for key in args.wellid2treatment.keys()}
    args.dosages = list(np.unique(list(args.wellid2dosage.values())))