import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np

def get_questionnaire_dataframe(worker_ids, questionnaires):
    df_list = []

    for idx, quest in zip(worker_ids, questionnaires):
        tmp = json.loads(quest)
        tmp["WorkerId"] = idx

        df_list.append(tmp)

    df = pd.DataFrame(df_list)
    df = df.set_index("WorkerId")
    df = df[~df.index.duplicated(keep='first')]

    return df


def get_sum_true_false(df, columns):
    return df.apply(lambda x: x[columns].values.sum(), axis=1)

def group_by_count(df, grouped_by, val, count):
    df_plot = df.groupby(grouped_by + [val]).sum().reset_index()
    df_grouped_by_control_hop = df.groupby(grouped_by).sum().reset_index()

    for row in df_grouped_by_control_hop.iterrows():
        tmp = row[1]

        tmp_filter = df_plot[grouped_by[0]] == tmp[grouped_by[0]]
        for i in grouped_by[1:]:
            tmp_filter = tmp_filter & (df_plot[i] == tmp[i])

        if tmp[count] == 0:
            df_plot.loc[tmp_filter, count] = 0
        else:
            df_plot.loc[tmp_filter, count] /= tmp[count]

    return df_plot


def big_plot_helper(df_diff_treatment, t_stack, df_diff_control, c_stack, val, ax, color, hatch,
                    map_val, set_title, y_ticks, show_y_ticks, x_ticks, show_x_ticks, y_label, show_y_label,
                    df_tag=None):
    if df_tag is not None:
        df_tag["style"] = "x"
        sns.lineplot(x="reduction", y="diff", ax=ax, style="style", legend=False,
                     data=df_tag, markers=True, dashes=False,
                     color=color, err_kws={"hatch": hatch, "alpha": 0.5})
    else:
        t_stack.append(df_diff_treatment[df_diff_treatment.val == val].groupby(["hop"]).mean().reset_index()["count"])
        c_stack.append(df_diff_control[df_diff_control.val == val].groupby(["hop"]).mean().reset_index()["count"])

        df_tmp = df_diff_control[df_diff_control.val == val]
        df_tmp["style"] = "x"
        line = sns.lineplot(x="reduction", y="diff", style="style", ax=ax, data=df_tmp, markers=True, dashes=False,
                            color=color, err_kws={"hatch": hatch, "alpha": 0.5, "edgecolor": "#030303"}, legend=False)

    plt.sca(ax)
    plt.xlabel("")
    plt.ylabel("")
    ax.plot([2, 5], [0, 0], ":", color="black", alpha=1, lw=1.5)

    # -- SET TITLE
    ax.set_title(map_val, fontsize=10) if set_title else ax.set_title("", fontsize=10)

    # -- SET YTICKS
    plt.yticks(y_ticks[0], y_ticks[1]) if show_y_ticks else \
        plt.yticks(y_ticks[0], [""] * len(y_ticks[1]))
    plt.ylim(y_ticks[2])

    # -- SET XTICKS
    plt.xticks(x_ticks[0], x_ticks[1]) if show_x_ticks else \
        plt.xticks(x_ticks[0], [""] * len(x_ticks[1]))
    plt.xlim(x_ticks[2])

    # SET YLABEL
    plt.ylabel(y_label) if show_y_label else None


def table_helper(t_stack, c_stack, row_labels, col_labels, number_cols, ax):
    tmp_t = [["{:.0%}".format(formatted_v) for formatted_v in v.values[2:]] for v in [t_stack[i]
                                                                                      for i in range(number_cols)]]
    tmp_c = [["{:.0%}".format(formatted_v) for formatted_v in v.values[2:]] for v in [c_stack[i]
                                                                                      for i in range(number_cols)]]
    tmp = []
    for t, c in zip(tmp_t, tmp_c):
        tmp.append(["{}/{}".format(v[0], v[1]) for v in zip(t, c)])
    result = tmp

    plt.sca(ax)

    plt.axis('off')
    the_table = plt.table(cellText=result, rowLabels=row_labels, colLabels=col_labels,
                          loc='center') if col_labels is not None else \
        plt.table(cellText=result, rowLabels=row_labels, loc='center')
    the_table.set_fontsize(12)
    the_table.scale(1, 1)

    return result


def compare_parts(root, src, dst, weights_categories_tag):
    src_occur = src["Tagging"].get_occurrences("Coarse", relative=False)
    dst_occur = dst["Tagging"].get_occurrences("Coarse", relative=False)

    diff = dict()
    top = 0
    bottom = 0

    for key in src_occur.keys():
        try:
            diff[key] = (dst_occur[key]) / src_occur[key]

            top += dst_occur[key] * weights_categories_tag[key]
            bottom += src_occur[key] * weights_categories_tag[key]
        except ZeroDivisionError:
            continue

    diff["All"] = top / bottom if bottom > 0 else None

    return diff


def pearsonr_ci(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def compare_parts_doggos(root, src, dst, weights_categories):
    src_occur = src["Doggos_crowd"]["Coarse"]
    dst_occur = dst["Doggos_crowd"]["Coarse"]

    diff = dict()
    top = 0
    bottom = 0
    for key in src_occur.keys():
        if len(src_occur[key]) == 0 or ("A" not in src_occur[key] and "B" not in src_occur[key]):
            continue
        tuples = list(zip(src_occur[key], dst_occur[key]))
        tmp_top = tuples.count(("A", "A")) + tuples.count(("A", "B")) + tuples.count(("B", "A")) + tuples.count(
            ("B", "B"))

        tmp_bottom = src_occur[key].count("A") + src_occur[key].count("B")
        diff[key] = tmp_top / tmp_bottom
        top += tmp_top * weights_categories[key]
        bottom += tmp_bottom * weights_categories[key]
    diff["All"] = top / bottom if bottom > 0 else None
    return diff


def qual_helper(x, hopper, hopper_control):
    try:
        return hopper.g.nodes[x]["Qualification"]
    except KeyError:
        return hopper_control.g.nodes[x]["Qualification"]


def edu_helper(x, hopper, hopper_control):
    mapping = {"Some College": "Less than College",
               "Some High School": "Less than College",
               "High School": "Less than College",
               "College": "College"}
    try:
        return mapping[hopper.g.nodes[x]["Education"]]
    except KeyError:
        return mapping[hopper_control.g.nodes[x]["Education"]]


def readibility_helper(x, fun, hopper, hopper_control):
    try:
        return fun(hopper.g.nodes[x]["Answer"])
    except KeyError:
        return fun(hopper_control.g.nodes[x]["Answer"])


def regplot_helper(color, df, axis, row=None, value=None):
    for idx, (c, ax) in enumerate(zip(color, axis)):
        df_tmp = df[df[row] == value[idx]] if value is not None else df

        sns.regplot(x="hop", y="val", x_estimator=np.mean,
                    data=df_tmp, color=c, ax=ax)

    if value is not None and len(value) == 2:
        x1 = df[df[row] == value[0]].hop.values
        x2 = df[df[row] == value[1]].hop.values
        y1 = df[df[row] == value[0]].val.values
        y2 = df[df[row] == value[1]].val.values


def helper_k(df_custom, k, category_helper, branch_to_topic, df, num_hops):
    tmp = []

    for key in category_helper["Fine"]:
        for branch in branch_to_topic.keys():
            df_tmp = df_custom[(df_custom.branch == branch) & (df_custom.hop == 1) & (df_custom.control == 0) & (
                    df_custom.key == key)].sort_values("val", ascending=False)
            ordered_fathers = df_tmp.node.values
            for hop in range(2, num_hops + 1):
                try:
                    to_remove = df_custom[
                        (df_custom.branch == branch) & (df_custom.hop == hop) & (df_custom.control == 0) & (
                                df_custom.father != ordered_fathers[k]) & (df_custom.key == key)]
                    tmp += list(zip(to_remove["node"].values, to_remove["key"].values))
                    assert len(to_remove) == 7
                except:
                    continue

                df_tmp = df_custom[(df_custom.branch == branch) & (df_custom.hop == hop) & (df_custom.control == 0) & (
                        df_custom.key == key)].sort_values("val", ascending=False)
                ordered_fathers = df_tmp.node.values

    df_t = df.set_index(["node", "key"])
    df_t.drop(tmp, inplace=True)
    df_t.reset_index(inplace=True)

    index_to_repeat = df_t.loc[(df_t.control == 0) & (df_t.hop > 1)].index.values
    index_test = df_t.loc[(df_t.control == 1) & (df_t.hop > 1)].index.values
    assert 8 * len(index_to_repeat) == len(index_test)
    index_other = df_t.loc[(df_t.control != 0) | (df_t.hop <= 1)].index.values
    df_t = df_t.reindex(np.concatenate([np.repeat(index_to_repeat, 8), index_other]))

    return df_t


def lines(axs):
    plt.sca(axs)
    axs.axhline(y=3 * 4.5 + 1, color="black", alpha=0.8)

    # axs.axhline(y=3 * 3.5 + 1, color="black", alpha=0.8)

    axs.axhline(y=3 * 7.5 + 1, color="black", alpha=0.8)

    axs.fill_between([0, 0.5], [-.5, -.5], [3 * 12.5 + 1, 3 * 12.5 + 1], facecolor="black", alpha=0.15)
    for v in np.arange(-0.5, 3 * 12.5 + 1, 3):
        axs.axhline(y=v, color="black", alpha=0.4)


def bars_plotter(df_tmp_fine, category_helper, val, colors_map, hierarchy_helper, ax,
                 helper_idy=lambda idy: idy * 3 + 1, alpha=0.5, marker="o", is_err=True):
    x = []
    y = []
    err = []
    for idy, category in enumerate(category_helper["Fine"][::-1]):
        if category == "Conclusion-general" and val == "Keyphrases":
            continue

        diff = df_tmp_fine[(df_tmp_fine.hop > 1) & (df_tmp_fine.key == category)]["diff"].values
        ci = sns.utils.ci(sns.algorithms.bootstrap(diff))
        mean = diff.mean()
        x.append(idy)
        y.append(ci[1] + ci[0] / 2)
        err.append(abs(ci[1] - ci[0]) / 2)
        if is_err:
            ax.errorbar([ci[1] + ci[0] / 2], [helper_idy(idy)], marker=marker, xerr=[abs(ci[1] - ci[0]) / 2],
                        color=colors_map[hierarchy_helper["Hierarchy_Mapping"]["Fine-Coarse"][category]],
                        capsize=5, alpha=alpha)
        else:
            ax.errorbar([ci[1] + ci[0] / 2], [helper_idy(idy)], marker=marker,
                        color=colors_map[hierarchy_helper["Hierarchy_Mapping"]["Fine-Coarse"][category]], alpha=alpha)

