import numpy as np
import re


def read_tensor_info(eom_obj, tei_infile, tamp_infile, eom_infile):
    # different 2e- integral, Tamp, and Ramp storage
    nocc = eom_obj.nocc
    eom_obj.tei = return_filled_tensor(tei_infile, 0, "p^ q^ r s",eom_obj)


    #print("2e- integral:",eom_obj.tei)
    #sys.exit()
    #eom_obj.t1amps = return_filled_tensor(tamp_infile, "p^ q", eom_obj.nocc,eom_obj)
    #eom_obj.t2amps = return_filled_tensor(tamp_infile, "p^ q r^ s", eom_obj.nocc,eom_obj)

    eom_obj.c1amps = return_filled_tensor(eom_infile, eom_obj.nocc, "p^ q", eom_obj)
    eom_obj.c2amps = return_filled_tensor(eom_infile, eom_obj.nocc, "p^ q r^ s", eom_obj)

    print(eom_obj.c1amps.shape,eom_obj.c2amps.shape)
    #sys.exit()
    # first transpose T2 and C2 from the current ordering a^ i b^ j to -1.0* a^ b^ i j
    #eom_obj.t2amps.transpose(0,2,1,3)
    #eom_obj.c2amps.transpose(0,2,1,3)

    #eom_obj.t2amps = -1.0*eom_obj.t2amps
    eom_obj.c2amps = -1.0*eom_obj.c2amps 

    # now expand the tensor 
    #eom_obj.t2amps = expanded_tensor(eom_obj,eom_obj.t2amps)
    eom_obj.c2amps = expanded_tensor(eom_obj,eom_obj.c2amps)


def expanded_tensor(eom_obj,tensor):
    print("tensor shape:", tensor.shape, tensor)
    v = eom_obj.nvirt
    o = eom_obj.nocc
    # Now fill the T2/C2 tensors out
    for a, b, i, j in ((a,b,i,j) for a in range(v) for b in range(a) for i in range(o) for j in range(i)):
        value = tensor[a,b,i,j]
        tensor[b,a,i,j]=-value
        tensor[a,b,j,i]=-value
        tensor[b,a,j,i]=value

    return tensor



def return_filled_tensor(filename, offset, index_pattern=None,eom_obj=None):
    """
    Read arbitrarily ranked tensor data from a file.

    Parameters
    ----------
    filename : str
        Input file.
    index_pattern : str or None
        Optional regex specifying which lines to parse.
        If None, parse any line ending in a float.

    Returns
    -------
    tensor : np.ndarray
        Dense tensor with rank inferred from input.
    """

    if offset == 0: 
        pattern = re.compile(
            r"""
            ^\s*
            (?P<indices>(?:\d+\^?\s+)+)
            (?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)
            \s*$
            """,
            re.VERBOSE
        )
    else:
        if len(index_pattern) == 9:
            pattern = re.compile(
                r"""
                ^\s*
                (?P<indices>(?:\d+\^\s+\d+\s*)+)
                \|\s*
                (?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)
                \s*$
                """,
                re.VERBOSE
            )
        else:
            pattern = re.compile(
                r"""
                ^\s*
                (?P<indices>\d+\^\s+\d+)
                \s*\|\s*
                (?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)
                \s*$
                """,
                re.VERBOSE
            )


    no = eom_obj.nocc
    nv = eom_obj.nvirt
    
    values = []
    op_strings=[]
    print(filename,pattern)
    with open(filename, "r") as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue
    
            index_tokens = match.group("indices").split()
            print(index_tokens) 
            indices = []
            for tok in index_tokens:
                idx = int(tok[:-1]) - offset if tok.endswith("^") else int(tok)
                if idx < 0:
                    raise ValueError(f"Invalid index {tok} with nocc={nocc}")
                indices.append(idx)
    
            value = float(match.group("value"))
            values.append(value)

            op_strings.append(tuple(indices))
            #print(indices,value)
            #entries.append((tuple(indices), value))

    if offset == 0:
        nbas = no + nv
        tensor = np.zeros((nbas,nbas,nbas,nbas))
    else:
        creation_index=[]
        for k, tok in enumerate(index_tokens):
            if tok.endswith("^"):
                creation_index.append(nv)
            else:
                creation_index.append(no)
    
        shape = tuple(creation_index)
    
        # infer tensor shape
        #shape = tuple(m + 1 for m in max_indices)
        tensor = np.zeros(shape)
 
    print("tensor shape is:",tensor.shape,offset)
    print(op_strings[2])
    # populate tensor
    for inds, val in zip(op_strings,values):
        tensor[inds] = val

    return tensor

