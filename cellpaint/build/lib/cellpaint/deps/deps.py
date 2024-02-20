class FoundItem(Exception):
    pass


def drop_numerically_unstable_cols(self, features):
    # check for numerical stability
    cols2drop = []
    for col in tqdm(features.columns[self.start_index:]):
        cond0 = (features[col].values > -1e-6)
        cond1 = (features[col].values < 1e-6)
        median = np.median(features[col].values)
        mad = stats.median_abs_deviation(features[col].values)
        if median < 1e-4:
            cols2drop.append(col)
        #     print(f"{col:50} {100*np.sum(cond)/len(features):.2f}")
        elif (100 * np.sum(cond0 & cond1) / len(features) > 10) or (np.abs(mad / median) < 1e-4):
            #         features.drop(col, axis=1, inplace=True)
            cols2drop.append(col)
        # else:
        #     features = features.loc[~(cond0 & cond1)]
    #     if np.sum(features[cond][col].values)>.1*len(features[col]):
    #         print(f"{col:50} {100*np.sum(cond)/len(features):.2f}")
    features.drop(cols2drop, axis=1, inplace=True)
    return features



def create_xlabels_above_figure_single_channel_dep(self):
    """adding text for each category of features (shape, intensity, moments, haralicks) for better
    visibility/readability"""
    xlocs = []
    xvals = ['Shape']
    # get the first moment key column index
    try:
        for cc in self.args.moment_keys:
            for it in self.fd_cols:
                if cc in it:
                    xlocs.append(self.fd_cols.index(it))
                    xvals.append(it)
                    raise FoundItem
    except:
        pass
    # get the first intensity key column index
    try:
        for cc in self.args.intensity_keys:
            for it in self.fd_cols:
                if cc in it:
                    xlocs.append(self.fd_cols.index(it))
                    xvals.append(it)
                    raise FoundItem
    except:
        pass

    # get haralick features column index
    haralick_names = []
    for cc in self.args.haralick_features:
        for it in self.fd_cols:
            if cc in it:
                xlocs.append(self.fd_cols.index(it))
                xvals.append(it)
                haralick_names.append(cc)
                break
    ############################################################################################
    # add vertical lines
    hh = list(np.arange(0, len(self.ylabels_left), 1))
    for x_ in xlocs:
        # x specifies where the line's starting point on the y-axis
        # xmin and xmax specify how long the line is (from where to where)
        self.heatmap.vlines(x=x_, ymin=0, ymax=len(hh), linestyle='dashed', linewidth=.4, color="black")
    ###############################################################################################
    xlocs.append(len(self.fd_cols))
    # add text for sub-categories of features above the figure object
    for beg, end, label in zip(xlocs[:-1], xlocs[1:], ["Shape", "Moments", "Intensity"] + haralick_names):
        # add some text to the center right of this band
        self.heatmap.text((beg + end) / 2, 1.05,
                          # '\n'.join(label.split())
                          label,
                          va='top', ha='center',
                          transform=self.heatmap.get_xaxis_transform(), size=14, )