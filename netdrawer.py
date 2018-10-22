# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/net_drawer.py
#
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#
#   $ dot -Tsvg my_output.dot -o my_output.svg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import defaultdict
import json
from onnx import ModelProto, GraphProto, NodeProto
import pydot  # type: ignore
from typing import Text, Any, Callable, Optional, Dict


OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}

BLOB_STYLE = {'shape': 'octagon'}

_NodeProducer = Callable[[NodeProto, int], pydot.Node]


def _escape_label(name):  # type: (Text) -> Text
    # json.dumps is poor man's escaping
    return json.dumps(name)


def _form_and_sanitize_docstring(s):  # type: (Text) -> Text
    url = 'javascript:alert('
    url += _escape_label(s).replace('"', '\'').replace('<', '').replace('>', '')
    url += ')'
    return url

timings = [ 1325706, 799440, 1263204, 257175, 936527, 2387987, 241757, 260674, 239340, 416096,
            422262, 101753, 1302205, 255840, 256091, 238423, 388095, 727271, 371094, 72168,
            1496044, 145171, 205172, 159504, 244174, 668436, 84169, 1276538, 145671, 222340,
            178588, 283175, 335177, 267424, 58919, 808940, 83086, 138254, 81336, 138670,
            390011, 57585, 3122175, 81586, 160588, 80085, 153421, 502681, 61502, 1581796,
            93753, 207673, 93669, 156505, 663186, 59502, 1563713, 93503, 202422, 94170,
            152255, 135504, 3697191, 240590, 294509, 347594
]

def GetOpNodeProducer(embed_docstring=False, **kwargs):  # type: (bool, **Any) -> _NodeProducer
    def ReallyGetOpNode(op, op_id):  # type: (NodeProto, int) -> pydot.Node
        if op.name:
            node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_id)
        else:
            print("No name operator, of type [{0},{1}]".format(op.op_type, op_id))
            if (op_id <= len(timings)):
                timing = timings[op_id]
            else:
                timing = 0
            node_name = '%s (op#%d) [%d ns]' % (op.op_type, op_id, timing)
        for i, input in enumerate(op.input):
            node_name += '\n input' + str(i) + ' ' + input
        for i, output in enumerate(op.output):
            node_name += '\n output' + str(i) + ' ' + output
        #node_name += '\n timestamp: '
        node = pydot.Node(node_name, **kwargs)
        if embed_docstring:
            url = _form_and_sanitize_docstring(op.doc_string)
            node.set_URL(url)
        return node
    return ReallyGetOpNode


def GetPydotGraph(
    graph,  # type: GraphProto
    name=None,  # type: Optional[Text]
    rankdir='LR',  # type: Text
    node_producer=None,  # type: Optional[_NodeProducer]
    embed_docstring=False,  # type: bool
):  # type: (...) -> pydot.Dot
    if node_producer is None:
        node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes = {}  # type: Dict[Text, pydot.Node]
    pydot_node_counts = defaultdict(int)  # type: Dict[Text, int]
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    _escape_label(
                        input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                    **BLOB_STYLE
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                _escape_label(
                    output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
                **BLOB_STYLE
            )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))
    return pydot_graph


def main():  # type: () -> None
    parser = argparse.ArgumentParser(description="ONNX net drawer")
    parser.add_argument(
        "--input",
        type=Text, required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "--output",
        type=Text, required=True,
        help="The output protobuf file.",
    )
    parser.add_argument(
        "--rankdir", type=Text, default='LR',
        help="The rank direction of the pydot graph.",
    )
    parser.add_argument(
        "--embed_docstring", action="store_true",
        help="Embed docstring as javascript alert. Useful for SVG format.",
    )
    args = parser.parse_args()
    model = ModelProto()
    with open(args.input, 'rb') as fid:
        content = fid.read()
        model.ParseFromString(content)
    pydot_graph = GetPydotGraph(
        model.graph,
        name=model.graph.name,
        rankdir=args.rankdir,
        node_producer=GetOpNodeProducer(
            embed_docstring=args.embed_docstring,
            **OP_STYLE
        ),
    )
    pydot_graph.write_dot(args.output)


if __name__ == '__main__':
    main()

# python netdrawer.py --input model.onnx --output model2.dot
# dot -Tsvg model2.dot -o model2.svg