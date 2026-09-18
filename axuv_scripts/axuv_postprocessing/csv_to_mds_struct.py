# csv_to_mds_struct.py
# This takes a csv file with tree definitions and adds it as a structure to the 
# WHAM tree


import MDSplus as mds
import csv
import re
import sys

source_file = 'AXUV_FIELDS.csv'

def strip_tree_qualifier(path):
    return re.sub(r'^\\[^:]+::', '\\\\', path)

def delete_node_recursive(tree,path):
    path = strip_tree_qualifier(path)
    node = tree.getNode(path)
    for child in node.getChildren():
        delete_node_recursive(tree, child.getFullPath())
    for member in node.getMembers():
        delete_node_recursive(tree, member.getFullPath())
    # Python tree.deleteNode can have some problems:
    mds.tcl(f'delete node "{path}"/confirm')

def delete_node_tcl(tree,path):
    expt = tree.tdiCompile('$EXPT').data()
    shot = tree.tdiCompile('$SHOT').data()
    mds.tcl(f'edit {expt}/shot={shot}')
    mds.tcl(f'delete node {path}/confirm')
    mds.tcl('write')

def auto_tagger(tree,verbose=True):
    '''
    Adds tags to the tree automatically for sub trees and structures that are two layers deep.
    Other tags need to be decided manually
    '''

    def autotag_children(node):
        for n in node.descendants:
            name = n.getName()
            #print(dir(n))
            if verbose:
                print(n)
                print(f'Is it a subtree? {n.subtree}')
                print(f'Number of children is : {n.getNumChildren()}')
                print(f'Current tags are {n.tags}')
            if not (name in n.tags) and (n.subtree or n.getNumChildren()):
                if verbose:
                    print(f'adding tag to {name}')
                n.addTag(f'{name}')
    top = tree.getNode('\\TOP')
    autotag_children(top)
    for n in top.descendants:
        autotag_children(n)
    tree.write()

def collapse_tree(tree,subtree,parent_node,shot,verbose=True):
    '''
    Collapses sub tree into a structure
    '''

    def putNode(tree,path_above,node):
        # Recursively populate a node structure
        path = f'{path_above}.{node.getName()}'
        tree.addNode(path,node.usage)
        print(path)
        if node.subtree:
            return
        for n in node.descendants:
            putNode(tree,path,n)
        #print(node.usage)
        #print(node.getData().decompile())
        #tree.getNode(path).putData(tree.tdiCompile(node.getData().decompile()))
        # print(node.data())
        # print(node.getData().decompile())
        # print(node.value_of())
        try:
            #print(node.getData().decompile())
            tree.getNode(path).putData(node.getData())
            print(path,tree.getNode(path).data())
        except:
            pass
        return

    # Verify that this is actually a subtree
    usage = tree.getNode(f'{parent_node}.{subtree}').usage
    if usage != 'SUBTREE':
        print(f'Not refactoring {subtree}, it is already not a subtree')
        return
    #n = tree.getNode(nodepath)
    # Get all of the subnodes
    st = mds.Tree(subtree,shot)
    stn = st.getNodeWild('***')
    stn = st.getNode('\\TOP').descendants
    #st_node = tree.getNode(f'{parent_node}.{subtree}')
    st_name = f'{parent_node}.{subtree}'
    #print(st_node)
    try:
        tree.deleteNode(st_name)
    except:
        pass
    print(st_name)
    tree.addNode(st_name,'STRUCTURE')
    for node in stn:
        putNode(tree,st_name,node)
    #print(n[0])
    tree.write()
    return

def write_node(tree,values,overwrite=False):
    # Get full path
    if values["Parent"] == '':
        fullpath = f'\\{values["Name"]}'
    else:
        fullpath = f'\\{values["Parent"]}.{values["Name"]}'
    # First check if node already exists
    # exist = True
    # try:
    #     tree.getNode(fullpath)
    # except:
    #     exist = False
    # if exist and not overwrite:
    #     print(f'WARNING - Not overwriting {fullpath} - Already exists')
    #     return
    # elif exist and overwrite:
    #     print('Deleting',fullpath)
    #     # Need to work in mdstcl for this?
    #     tree.write()
    #     delete_node_tcl(tree,fullpath)
    #     #delete_node_recursive(tree,fullpath)
    #     # Reopen the tree if doing this
    #     tree = mds.Tree(tree.tdiCompile('$EXPT').data(),tree.tdiCompile('$SHOT').data(),'edit')

    # Now add the node:
    # check if node exists, if it does, do nothing
    if not overwrite:
        try:
            n = tree.getNode(fullpath)
            print(f'{fullpath} exists, not overwriting')
            return
        except:
            pass
    print(values)
    tree.addNode(fullpath,usage=values['Usage'])
    node = tree.getNode(fullpath)

    if values['Tags'] != '':
        tags = eval(values['Tags'])
        for tag in tags:
            node.addTag(tag)

    if values['Value'] != '':
        expr = values['Value']
        if values['Units'] != '':
            units = values['Units']
            expr = f'BUILD_WITH_UNITS({expr},"{units}")'
        print(expr)
        node.putData(tree.tdiCompile(expr))
        
    
def check_and_delete_node(expt,shot,values):
    tree = mds.Tree(expt,shot,'edit')
    if values["Parent"] == '':
        fullpath = f'\\{values["Name"]}'
    else:
        fullpath = f'\\{values["Parent"]}.{values["Name"]}'
    exist = True
    try:
        tree.getNode(fullpath)
    except:
        exist = False
    if exist:
        delete_node_tcl(tree,fullpath)
    return

def csv_to_mds_struct(expt,source_file,shot=-1,overwrite=False):

    tree = mds.Tree(expt,shot,'edit')

    # if overwrite, loop twice.  First to delete the nodes and second to add them
    if overwrite:
        collapse_tree(tree,'DIAG','\\TOP',shot)
        with open(source_file,mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                check_and_delete_node(expt,shot,row)
        tree = mds.Tree(expt,shot,'edit')

    # add some tags
    #auto_tagger(tree)
    # Add tags to the ACQ1001s....auto-tagger above is barfing on some of the devices in RAW
    n = tree.getNode('\\WHAM::RAW.ACQ1001_639')
    name = n.getName()
    if not (name in n.tags):
        n.addTag(f'{name}') 

    with open(source_file,mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            write_node(tree,row,overwrite)
            fullpath = f'\\{row["Parent"]}.{row["Name"]}'
            #break
    tree.write()
    #tree.cleanDatafile()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        shot = int(sys.argv[1])
    else:
        shot = -1
    csv_to_mds_struct(expt='WHAM',source_file=source_file,shot=shot,overwrite=False)

