# set up the calculate feature importance function
def calc_feature_importance(descriptors, tstats):
    """
    Calculate feature importance from a fitted model
    :param descriptors: list of descriptors
    :param tstats: list of t-statistics
    :return: dictionary of feature importance
    """

    feature_importance = {a: b for a, b in zip(descriptors, tstats) if not np.isnan(b)}
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
    )
    # Fitted space important features
    return feature_importance

def get_beta (space, desc_type: str, return_beta, return_examples, radius, similarity_threshold):
    """
    Still need to document this function
    """
    
    beta = exmol.lime_explain(space, desc_type, return_beta, return_examples, radius, similarity_threshold)
    if desc_type == "ecfp":
        exmol.plot_descriptors(space, radius = radius, output_file=f"{desc_type}.svg")
        # plt.close() is used to close the figure window because
        # it seems not to be needed here, but is nevertheless saled to file
        # I just commented out this line from the original code to see what it was doing
        # plt.close()
    else:
        exmol.plot_descriptors(space, radius = radius)
    return beta

#lime explain
def ind_explainers(space, desc_type, return_beta: bool, return_examples: bool, radius, similarity_threshold):
    desc_type = ["Classic",
                  "maccs",
                     "ecfp", "chromophore", "expanded_chromophore"]
    #descriptor type list
    for d in desc_type:
        beta = exmol.lime_explain(space, descriptor_type=d, return_beta = return_beta, return_examples = return_examples, radius = radius, similarity_threshold = similarity_threshold)
        if d == "ecfp":
            exmol.plot_descriptors(space, radius, output_file=f"{d}.svg")
            # plt.close() is used to close the figure window because
            # it seems not to be needed here, but is nevertheless saled to file
            # I just commented out this line from the original code to see what it was doing
            # plt.close()
        else:
            exmol.plot_descriptors(space, radius)



# desc_type = ["Classic", "ecfp", "maccs"]
# The descriptor type that remains with the full "space" is the one that is done last.
# This would be the one that shows up on the parity plot later on.
# desc_type = ["Classic", "maccs", "ecfp"]
#descriptor type list


# This allows us to do a parity plot



def plot_parity (space, beta, similarity_threshold):
    fkw = {"figsize": (6, 4)}
    font = {"family": "normal", "weight": "normal", "size": 16}

    fig = plt.figure(figsize=(10, 5))
    mpl.rc("axes", titlesize=12)
    mpl.rc("font", size=16)
    ax_dict = fig.subplot_mosaic("AABBB")
    
    # Plot space by fit
    svg = exmol.plot_utils.plot_space_by_fit(
        space,
        [space[0]],
        figure_kwargs=fkw,
        mol_size=(200, 200),
        offset=1,
        ax=ax_dict["B"],
        beta=beta,
    )
    # Compute y_wls
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in space])
    # print (w)
    non_zero = w > 10 ** (similarity_threshold)
    # show the number of true values in non_zero
    print (len(non_zero[non_zero == True]))

    # print (non_zero)
    w = w[non_zero]
    # this just resets the w values to only be the ones above threshold
    N = w.shape[0]
    # ys is just getting the previously predicted y values for all those molecules that are above threshold for similarity
    ys = np.array([e.yhat for e in space])[non_zero].reshape(N).astype(float)

    # x_mat is just getting the descriptors for all those molecules that are above threshold for similarity
    x_mat = np.array([list(e.descriptors.descriptors) for e in space])[non_zero].reshape(
        N, -1
    )
    # the descriptors ion x_mat are the features pulled from the most recently added descrioptor type in the space
    # beta is the weights for the features in x_mat
    # the @ is the matrix multiplication operator.
    print (f"x_mat shape is {x_mat.shape}")
    print (f"beta shape is {beta.shape}")
    y_wls = x_mat @ beta
    print (f"y_wls shape is {y_wls.shape}")
    # y_wls is added to the mean of ys because 
    y_wls += np.mean(ys)

    #lower and higher used to set plot limits
    lower = np.min(ys)
    higher = np.max(ys)

    # set transparency using w, i.e. the color will change based on the similarity of the molecule to the query molecule
    norm = plt.Normalize(min(w), max(w))
    cmap = plt.cm.Oranges(w)
    cmap[:, -1] = w

    # so when simiarity is high, w is close to 1
    def weighted_mean(x, w):
        return np.sum(x * w) / np.sum(w)


    def weighted_cov(x, y, w):
        return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)


    def weighted_correlation(x, y, w):
        return weighted_cov(x, y, w) / np.sqrt(
            weighted_cov(x, x, w) * weighted_cov(y, y, w)
        )

    # this just allows a weighted correlation to be calculated
    corr = weighted_correlation(ys, y_wls, w)

    ax_dict["A"].plot(
        np.linspace(lower, higher, 100), np.linspace(lower, higher, 100), "--", linewidth=2
    )
    sc = ax_dict["A"].scatter(ys, y_wls, s=50, marker=".", c=cmap, cmap=cmap)
    ax_dict["A"].text(max(ys) - 2, min(ys) + 0.5, f"weighted \ncorrelation = {corr:.3f}")
    ax_dict["A"].set_xlabel(r"$\hat{y}$")
    ax_dict["A"].set_ylabel(r"$g$")
    ax_dict["A"].set_title("Weighted Least Squares Fit")
    ax_dict["A"].set_xlim(lower, higher)
    ax_dict["A"].set_ylim(lower, higher)
    ax_dict["A"].set_aspect(1.0 / ax_dict["A"].get_data_ratio(), adjustable="box")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=norm)
    cbar = plt.colorbar(sm, orientation="horizontal", pad=0.15, ax=ax_dict["A"])
    cbar.set_label("Chemical similarity")
    plt.tight_layout()
    plt.savefig("weighted_fit_3_1_2023.svg", dpi=300, bbox_inches="tight", transparent=False)



    # simple descriptors plot
def simple_lime_explain (space, desc_type, return_beta: bool, return_examples:bool, radius, similarity_threshold):
        
    exmol.lime_explain(space, descriptor_type=desc_type, return_beta = return_beta, return_examples = return_examples, radius = radius, similarity_threshold=similarity_threshold)
    wls_attr = calc_feature_importance(list(space[0].descriptors.descriptor_names), list(space[0].descriptors.tstats))
    wls_attr

    x = wls_attr.keys()
    xaxis = np.arange(len(x))
    x_colors = ["purple" if t in names else "black" for t in x]

    rf_imp = {a: b for a, b in zip(names, loaded_clf.feature_importances_)}
    rf_x = np.zeros(len(x))
    rf_y = np.zeros(len(x))
    for i, j in enumerate(x):
        if j in rf_imp:
            rf_x[i] = i
            rf_y[i] = rf_imp[j]

    width = [wls_attr[i] for i in x]
    colors = ["#F06060" if i < 0 else "#1BBC9B" for i in width]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(xaxis + 0.2, width, 0.75, label="WLS", color=colors)

    plt.xticks(fontsize=12)
    plt.xlabel("Feature t-statistics", fontsize=12)
    plt.yticks(xaxis, x, fontsize=12)
    [t.set_color(i) for (i, t) in zip(x_colors, ax.yaxis.get_ticklabels())]
    plt.gca().invert_yaxis()
    plt.title("Random Forest Regression", fontsize=12)

