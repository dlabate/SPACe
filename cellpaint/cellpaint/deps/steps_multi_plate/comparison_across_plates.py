from pathlib import WindowsPath

import numpy as np
import pandas as pd
from cellpaint.steps_single_plate.step0_args import Args
from cellpaint.steps_single_plate.step8_rocauc import ROCAUCAnalysis


class DMSOAcrossPlates(ROCAUCAnalysis):
    plot_type = 4
    is_multi_dose = False

    def __init__(self, cell_line, index):
        args = Args(
            experiment=f"20230411-CP-MBolt-FlavScreen-UMUC3-1-3_20230412_190700",
            hard_drive='E',
            cellpaint_folder="CellPainting").args
        ROCAUCAnalysis.__init__(self, args)
        self.main_path = WindowsPath(r"P:\tmp\MBolt\Cellpaint-Flavonoid")
        # self.main_path = WindowsPath(r"E:\CellPainting")

        self.index = index
        self.cell_line = cell_line.lower()
        self.density = self.args.densities[0]
        self.distmap_filename = f"density={self.density}_cellline={self.cell_line}_2_wasserstein.csv"
        self.exp_paths = list(self.main_path.iterdir())
        self.exp_paths = list(filter(lambda x: self.cell_line in str(x).lower(), self.exp_paths))
        self.save_path = self.main_path/f"DMSO-Comparison-Across-Plates"
        self.save_path.mkdir(exist_ok=True, parents=True)

    def calc(self, ):
        distmap = []
        for exp_path in self.exp_paths:
            csv_file_path = exp_path/"Step5_DistanceMaps"/self.distmap_filename
            print(csv_file_path)
            # loading csv file
            tmp = pd.read_csv(csv_file_path)
            tmp = tmp.loc[(tmp["well-status"] == "pass") & (tmp["treatment"] == "dmso")]
            # This is a must for the rest to work properly!!!!
            tmp.reset_index(inplace=True, drop=True)
            distmap.append(tmp)
            del tmp
        distmap = pd.concat(distmap, ignore_index=True, axis=0)
        # treatment level unique values from distmap
        unix = distmap.groupby(self.part_2_group_cols)["cell-count"].agg(self.aggs).reset_index()
        unix.rename(dict(zip(self.aggs, self.aggs_new)), axis=1, inplace=True)
        distmap[self.hit_col] = 0
        unix[self.hit_col] = 0
        print(unix)
        roc_curves, roc_auc = self.compute_roccurve_and_rocauc(distmap, distmap, self.plot_type)
        assert roc_curves.shape[2] == roc_auc.shape[2] == distmap.shape[0]
        # calculate roc_curves, and roc_aucs for visualization as well as hit calling
        roc_curves_median, roc_auc_median, roc_curves_mad, roc_auc_mad = \
            self.get_median_and_mad_maps_per_well(distmap, unix, roc_curves, roc_auc)

        title1 = f"l1-Norm ROC-Curves-Median: Density={self.density}  Cell-line={self.cell_line}"
        title2 = f"l1-Norm ROC-AUC-Median-Heatmap: Density={self.density}  Cell-line={self.cell_line}"
        title3 = f"l1-Norm ROC-Curves-MAD: Density={self.density}  Cell-line={self.cell_line}"
        title4 = f"l1-Norm ROC-AUC-MAD-Heatmap: Density={self.density}  Cell-line={self.cell_line}"

        print(f"Density={self.density}  Cell-line={self.cell_line}")
        savename1 = f"c{self.index}00-Density={self.density}-Cellline={self.cell_line}_Median_ROC-Curves-l1"
        savename2 = f"c{self.index}10-Density={self.density}-Cellline={self.cell_line}_Median_ROC-AUCs-l1"
        savename3 = f"c{self.index}01-Density={self.density}-Cellline={self.cell_line}_MAD_ROC-Curves-l1"
        savename4 = f"c{self.index}11-Density={self.density}-Cellline={self.cell_line}_MAD_ROC-AUCs-l1"

        self.plot_roccurves(
            roc_curves_median[:, :, :, :],
            unix,
            self.plot_type,
            self.is_multi_dose,
            title=title1,
            savename=savename1,)
        self.plot_rocauc_heatmap(
            roc_auc_median[:, :, :],
            unix,
            self.aggs_new[0],
            self.plot_type,
            self.is_multi_dose,
            title=title2,
            savename=savename2)

        self.plot_roccurves(
            roc_curves_mad[:, :, :, :],
            unix,
            self.plot_type,
            self.is_multi_dose,
            title=title3,
            savename=savename3,)
        self.plot_rocauc_heatmap(
            roc_auc_mad[:, :, :],
            unix,
            self.aggs_new[0],
            self.plot_type,
            self.is_multi_dose,
            title=title4,
            savename=savename4)


def main():
    for ii, cell_line in enumerate(["UMUC3", "RT4", "5637"]):
        myclass = DMSOAcrossPlates(cell_line=cell_line, index=ii)
        myclass.calc()


if __name__ == "__main__":
    main()
