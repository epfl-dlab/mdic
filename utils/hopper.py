import networkx as nx
import pandas as pd
import collections


class Hopper:

    def __init__(self):
        self.g = nx.DiGraph()
        self.num_hops = 0

    """ ==========
    |   Helpers
    =========="""

    def create_node(self, node_id, level=None, branch=None, **kwargs):
        """ This simply creates a node specifying a level and a branch

        :param node_id: Id of the node to be created.
        :param level: Level of the node to be created.
        :param branch: Branch of the node to be created.
        :param kwargs: Rest of the arguments of the node.
        :return: Nothing.
        """

        if level is None:
            level = 0
        if branch is None:
            branch = node_id

        kwargs["level"] = level
        kwargs["branch"] = branch

        if level > self.num_hops:
            self.num_hops = level

        self.g.add_node(node_id, **kwargs)

    def create_edge(self, src, dst, **kwargs):
        """ Creates an edge without having to access the inner attribute g.
        
        :param src: Source of the edge.
        :param dst: Destination of the edge.
        :param kwargs: Other attributes of the edge.
        :return: Nothing.
        """

        self.g.add_edge(src, dst, **kwargs)

    """ ==========
    |   Mappers
    =========="""

    def compare_with_root(self, f, f_root=None, branches=None):
        """ Given a function f
                f: node x root --> object
            and a function f_root
                f_root: root --> object
            creates a map:

            {
                "branch_1": {
                        0: {
                            node_id_1: object,
                            ...
                            },
                        ...
                    }
                ...
            }

        :param f: Function. To be applied at every node.
        :param f_root: Function. To be applied at the root.
        :param branches: Iterable. Only returns the branches specified here. If none, select all branches.
        :return: Dictionary.
        """

        roots_filter = lambda n: n["level"] == 0 if branches is None else \
            lambda n: n["level"] == 0 and n["branch"] in branches

        roots = {key: node for key, node in self.g.nodes(data=True) if roots_filter(node)}

        other = [(key, node) for key, node in self.g.nodes(data=True) if not roots_filter(node)]

        output = {key: dict() for key in roots.keys()}

        for key, node in other:
            if node["level"] not in output[node["branch"]]:
                output[node["branch"]][node["level"]] = dict()

            result = f(node, roots[node["branch"]])
            output[node["branch"]][node["level"]][key] = result

        if f_root is not None:
            for key, node in roots.items():
                result = f_root(node)
                output[key][0] = {key: result}

        return output

    def compare_with_previous(self, f, branches=None):
        """ Given a function f
                f: root x source x dst --> dictionary
            creates a map:
            {
                1: [(src, dst, branch, dictionary ), ...]
                ...
            }

        :param f: Function. To be applied at every edge.
        :param branches: Iterable. Only returns the branches specified here. If none, select all branches.
        :return: Dictionary.
        """

        output = dict()

        for src, dst in self.g.edges():
            src_node = self.g.nodes[src]
            dst_node = self.g.nodes[dst]

            if branches is not None and src_node["branch"] not in branches:
                continue

            if dst_node["level"] not in output:
                output[dst_node["level"]] = []

            root = self.g.nodes[src_node["branch"]]

            result = f(root, src_node, dst_node)

            output[dst_node["level"]].append((src, dst, src_node["branch"], result))

        return output

    """ ==========
    | Data Frames
    =========="""

    def get_count_comparison_tree_dataframe(self, f, f_root=None, branches=None):
        """ Given a function f
                f: node x root --> dictionary
            and a function f_root
                f_root: root --> dictionary
            creates a map:

            {
                "branch_1": {
                        "category_1": {
                            "token_1": {
                                "hop_0": (#num_occur, % occur)
                            ]
                        }
                        ...
                    }
                ...
            }

            It then uses this to create a dataframe:

                    branch          category        hop	intensity	word
                0	NEJMoa1112010	Participants	0	1.000000	stroke
                1	NEJMoa1112010	Participants	1	1.000000	stroke

        :param f: Function. To be applied at every node.
        :param f_root: Function. To be applied at the root.
        :param branches: Iterable. Only returns the branches specified here. If none, select all branches.
        :return: Dataframe.
        """
        comparison_tree = self.compare_with_root(f, f_root, branches)

        count_comparison_tree = dict()

        # Creates the tree

        for branch, list_texts in comparison_tree.items():
            count_comparison_tree[branch] = dict()
            text_dict = list_texts[0]
            for name, category_dict in text_dict.items():
                for category, values in category_dict.items():
                    count_comparison_tree[branch][category] = dict()
                    for value in values:
                        count_comparison_tree[branch][category][value] = dict()
                        for hop in range(self.num_hops+1):
                            count_comparison_tree[branch][category][value][hop] = [0, 0]

        # Populates the tree

        for branch, list_texts in comparison_tree.items():
            for hop, text_dict in list_texts.items():
                num_papers_hop = len(text_dict)
                for name, category_dict in text_dict.items():
                    for category, values in category_dict.items():
                        for value in values:
                            count_comparison_tree[branch][category][value][hop][0] += 1
                            count_comparison_tree[branch][category][value][hop][1] += 1 / num_papers_hop

        # Creates dataframe
        tmp = []
        for branch, items1 in count_comparison_tree.items():
            for category, items2 in items1.items():
                for word, items3 in items2.items():
                    for hop, intensity in items3.items():
                        tmp.append(
                            {"branch": branch, "category": category, "word": word, "hop": hop,
                             "intensity": intensity[0], "norm_intensity": intensity[1]})
        df_count = pd.DataFrame(tmp)

        return df_count

    def get_tagging_dataframe(self, f, f_root=None, branches=None, norm_enabled=True):
        """ Given a function f
                f: node x root --> dictionary
            and a function f_root
                f_root: root --> dictionary
            creates a dataframe:
                    branch	        hop	  key	val   val_norm
                0	NEJMoa1112010	1	  Sex	3.00  1.00
                1	NEJMoa1112010	2	  Age	2.00  0.66
            where val is the key and the value are the keys and values returned by the function.

            The field val_norm emerges when the parameter `norm_enabled` is true and when the values in the dictionary
            are all integers. val_norm is normalized with the corresponding value in the root.

        :param f: Function. To be applied at every node.
        :param f_root: Function. To be applied at the root.
        :param branches: Iterable. Only returns the branches specified here. If none, select all branches.
        :param norm_enabled: Boolean. True if you want to get the extra field.
        :return: Dataframe.
        """

        a = self.compare_with_root(f, f_root, branches)

        tmp = []

        for branch, items1 in a.items():
            for hop, key_value in items1.items():
                for k2, v2 in key_value.items():
                    for key_cat, val_cat in v2.items():
                        if norm_enabled:
                            norm = a[branch][0][branch][key_cat]

                            if norm == 0:
                                val_norm = 0
                            else:
                                val_norm = val_cat / norm
                            row_tmp = {"hop": hop, "node": k2, "branch": branch, "key": key_cat,
                                       "val": val_cat, "val_norm": val_norm}
                        else:
                            row_tmp = {"hop": hop, "node": k2, "branch": branch, "key": key_cat, "val": val_cat}
                        tmp.append(row_tmp)

        return pd.DataFrame(tmp)

    def get_tagging_dataframe_list(self, f, hierarchy_helper=None, f_root=None,
                                   branches=None, val_default=("A", "B", "C", "D"), output_nodes=False):
        """ Given a function f
                f: node x root --> dictionary of lists of categories
            and a function f_root
                f_root: root --> dictionary of lists of categories
            creates a dataframe:
                    branch          hop   key     val   norm
                0   NEJMoa1112010   1     Sex     A     1.00
                1   NEJMoa1112010   2     Age     B     0.80

        :param f: Function. To be applied at every node.
        :param hierarchy_helper: This is a mapping between coarser grained categories and finer grain categories
        :param f_root: Function. To be applied at the root.
        :param branches: Iterable. Only returns the branches specified here. If none, select all branches.
        :param norm_enabled: Boolean. True if you want to get the extra field.
        :param val_default:
        :return: Dataframe.
        """

        a = self.compare_with_root(f, f_root, branches)

        tmp = []

        for branch, items1 in a.items():
            for hop, key_value in items1.items():
                for k2, v2 in key_value.items():
                    for key_cat, list_cat in v2.items():
                        for val_cat in list_cat:
                            if hierarchy_helper is not None:
                                row_tmp = {"hop": hop, "branch": branch, "count": 1, "key": key_cat, "val": val_cat,
                                           "key_coarse": hierarchy_helper["Hierarchy_Mapping"]["Fine-Coarse"][key_cat]}
                            else:
                                row_tmp = {"hop": hop, "branch": branch, "count": 1, "key": key_cat, "val": val_cat}
                            if output_nodes:
                                row_tmp["node"] = k2

                            tmp.append(row_tmp)

                        for val_cat in set(val_default) - set(list_cat):
                            if hierarchy_helper is not None:
                                row_tmp = {"hop": hop, "branch": branch, "count": 0, "key": key_cat, "val": val_cat,
                                           "key_coarse": hierarchy_helper["Hierarchy_Mapping"]["Fine-Coarse"][key_cat]}
                            else:
                                row_tmp = {"hop": hop, "branch": branch, "count": 0, "key": key_cat, "val": val_cat}
                            if output_nodes:
                                row_tmp["node"] = k2
                            tmp.append(row_tmp)

        return pd.DataFrame(tmp)

    def get_tagging_hop_dataframe(self, compare_parts):
        """ Given a function f
                f: root x source x dst --> dictionary
            creates a dataframe

        :param compare_parts:
        :return:
        """

        v = self.compare_with_previous(compare_parts)

        tmp = []

        for hop, edges in v.items():
            for src, dst, branch, val in edges:
                for key_cat, val_cat in val.items():
                    row_tmp = {"hop": hop, "src": src, "dst": dst, "branch": branch, "key": key_cat, "val": val_cat}
                    tmp.append(row_tmp)

        return pd.DataFrame(tmp)

    @staticmethod
    def from_graph(graph, num_hops):
        hopper = Hopper()
        hopper.g = graph
        hopper.num_hops = num_hops
        return hopper
