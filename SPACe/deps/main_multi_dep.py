from cellpaint.utils.args import CellPaintArgs
from cellpaint.utils.step4_get_distance_maps import DRCBMCellLinesAcrossExperiments, DMSOBM
from cellpaint.utils.step5_plot_heatmaps import stepV_main_run_loop


def stepIV(group, remove_outer_wells):
    """
    First Run _main.py to finish the first 3 steps_single_plate for each experiment group.
    Then, get distance feature maps for groups of experiments.
    """
    start_time = time.time()
    args = CellPaintArgs(
        experiment=group["anchor_experiment"],
        mode="full",
    ).args
    args.remove_outer_wells = remove_outer_wells

    if len(args.dosages) > 1:  # case 1

        print("Detect multiple Dosages ...\n"
              "Creating Distance Maps for a Dosage Response Benchmarking ...")
        distmap = DRCBMCellLinesAcrossExperiments(args, )
        distmap.calculate(group["experiments"], group["anchor_experiment"])

    if len(args.celllines) > 1:
        distmap = DMSOBM(args)
        for (anchor_experiment, anchor_cellline) in group["anchor_celllines"]:
            distmap.calculate(group["experiments"], anchor_experiment, anchor_cellline)

    print(f"prgogram finished analyzing experiment "
          f"in {(time.time()-start_time)/3600} hours...")
    print('\n')


def stepV(group):
    for item in group["experiments"]:
        print(item)
        args = CellPaintArgs(
            experiment=item,
            mode="full",
        ).args

        stepV_main_run_loop(args)


def main_worker(group, remove_outer_wells):
    stepIV(group, remove_outer_wells)
    stepV(group)
    print('\n')


if __name__ == "__main__":

    seema1 = {
        "experiments": ["20220920-CP-Bolt-Seema",],
        "anchor_experiment": "20220920-CP-Bolt-Seema",
        "anchor_celllines": [("20220920-CP-Bolt-Seema", "ht29-scr")]
    }
    seema2 = {
        "experiments": ["20220930-CP-Bolt-Seema", ],
        "anchor_experiment": "20220930-CP-Bolt-Seema",
        "anchor_celllines": [("20220930-CP-Bolt-Seema", "ht29-scr")]
    }
    seema3 = {
        "experiments": ["20221021-CP-Bolt-Seema", ],
        "anchor_experiment": "20221021-CP-Bolt-Seema",
        "anchor_celllines": [("20221021-CP-Bolt-Seema", "ht29-scr")]
    }
    ##################################################################

    seema5 = {
        "experiments": ["20230112-CP-Bolt-Seema", ],
        "anchor_experiment": "20230112-CP-Bolt-Seema",
        "anchor_celllines": [("20230112-CP-Bolt-Seema", "ht29-scr")]
    }

    seema6 = {
        "experiments": ["20230116-CP-Bolt-Seema", ],
        "anchor_experiment": "20230116-CP-Bolt-Seema",
        "anchor_celllines": [("20230116-CP-Bolt-Seema", "ht29-scr")]
    }

    seema7 = {
        "experiments": ["20230120-CP-Bolt-Seema", ],
        "anchor_experiment": "20230120-CP-Bolt-Seema",
        "anchor_celllines": [("20230120-CP-Bolt-Seema", "ht29-scr")]
    }

    seema8 = {
        "experiments": ["20230124-CP-Bolt-Seema", ],
        "anchor_experiment": "20230124-CP-Bolt-Seema",
        "anchor_celllines": [("20230124-CP-Bolt-Seema", "ht29-scr")]
    }

    ##############################################################################
    mike_flavonoid1 = {
        "experiments": ["20220912-CP-Bolt-MCF7", ],
        "anchor_experiment": "20220912-CP-Bolt-MCF7",
        "anchor_celllines": [("20220912-CP-Bolt-MCF7", "mcf7")]
    }

    mike_flavonoid2 = {
        "experiments": ["20220929-CP-Bolt-MCF7", ],
        "anchor_experiment": "20220929-CP-Bolt-MCF7",
        "anchor_celllines": [("20220929-CP-Bolt-MCF7", "mcf7")]
    }

    mike_flavonoid3 = {
        "experiments": ["20221024-CP-Bolt-MCF7", ],
        "anchor_experiment": "20221024-CP-Bolt-MCF7",
        "anchor_celllines": [("20221024-CP-Bolt-MCF7", "mcf7")]
    }
    #############################################################################################
    fabio_drc1 = {
        "experiments": ["20220831-CP-Fabio-DRC-BM-R01", "20220908-CP-Fabio-DRC-BM-R02"],
        "anchor_experiment": "20220831-CP-Fabio-DRC-BM-R01",
        "anchor_celllines": [("20220831-CP-Fabio-DRC-BM-R01", "u2os")]
    }

    fabio_drc2 = {
        "experiments": ["20221102-CP-Fabio-DRC-BM-P01", "20221102-CP-Fabio-DRC-BM-P02"],
        "anchor_experiment": "20221102-CP-Fabio-DRC-BM-P01",
        "anchor_celllines": [("20221102-CP-Fabio-DRC-BM-P01", "mcf10a"),
                             ("20221102-CP-Fabio-DRC-BM-P02", "u2os"), ]
    }

    fabio_drc3 = {
        "experiments": ["20221109-CP-Fabio-DRC-BM-P01", "20221109-CP-Fabio-DRC-BM-P02"],
        "anchor_experiment": "20221109-CP-Fabio-DRC-BM-P01",
        "anchor_celllines": [("20221109-CP-Fabio-DRC-BM-P01", "mcf10a"),
                             ("20221109-CP-Fabio-DRC-BM-P02", "u2os"), ]
    }

    fabio_drc4 = {
        "experiments": ["20221116-CP-Fabio-DRC-BM-P01", "20221116-CP-Fabio-DRC-BM-P02",],
        "anchor_experiment": "20221116-CP-Fabio-DRC-BM-P01",
        "anchor_celllines": [("20221116-CP-Fabio-DRC-BM-P01", "mcf10a"),
                             ("20221116-CP-Fabio-DRC-BM-P02", "u2os"), ]
    }

    chris_drc1 = {
         "experiments": ["20221207-CP-CCandler-Exp2244-1", "20221208-CP-CCandler-Exp2244-2", ],
         "anchor_experiment": "20221207-CP-CCandler-Exp2244-1",
         "anchor_celllines": [("20221207-CP-CCandler-Exp2244-1", "mcf10a"),
                              ("20221208-CP-CCandler-Exp2244-2", "u2os"), ]
    }
    ###################################################################
    mike_bladder_1 = {
         "experiments": ["20230119-CP-Bolt-Bladder", ],
         "anchor_experiment": "20230119-CP-Bolt-Bladder",
         "anchor_celllines": [("20230119-CP-Bolt-Bladder", "5637"), ]
    }

    # entry point of the program is creating the necessary args
    for grp in [
        # seema1,
        # seema2,
        # seema3,

        # mike_flavonoid1,
        # mike_flavonoid2,
        # mike_flavonoid3,

        # fabio_drc1,
        # fabio_drc2,
        # fabio_drc3,
        # fabio_drc4,

        # chris_drc1,
        # seema5,
        # seema6,
        # seema7,
        # seema8,
        mike_bladder_1
    ]:
        main_worker(grp, remove_outer_wells=False)
        print('\n')