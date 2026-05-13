import numpy as np
import re

def read_r1_r2(eom_obj,tamp_infile):
    t2amp={}
    t1amp={}

    read_amps=False
    nv = np.shape(eom_obj.c1amps)[0]
    no = np.shape(eom_obj.c1amps)[1]
    c1 = np.zeros((nv,no))
    c2 = np.zeros((nv,no,nv,no))
    print('reading tamp file:',tamp_infile)
    with open(tamp_infile,'r') as f:
        for line in f:
            if read_amps:#have to uncomment 109,13 for nospace
                #amp_key=float(line.split()[-1])#.strip('|'))
                amp_key=float(line.split()[-1].strip('|'))
                #index_list=line.split()
                index_list=line.split()[:-1]
                index_list.append('|')
                index_list.append(amp_key)
                operator_list=[]
                for operator in range(6): # max T2, min T1
                    if index_list[operator] == '|':
                        break
                    operator_list.append(int(index_list[operator].strip('^')))
                print('op list:',operator_list,'amp key:',amp_key)
                if len(operator_list)==4: #dealing with t2amp
                    t2amp.update({amp_key:operator_list})
                    a=operator_list[0]-eom_obj.nocc
                    b=operator_list[2]-eom_obj.nocc
                    i=operator_list[1]
                    j=operator_list[3]
                    print('op list:',operator_list[0],operator_list[1],operator_list[2],operator_list[3])
                    print('t2 dim:',np.shape(eom_obj.c2amps))
                    c2[a,i,b,j]=amp_key
                    c2[b,i,a,j]= -1.0* amp_key
                    c2[a,j,b,i]= -1.0*amp_key
                    c2[b,j,a,i]=amp_key

                    #sys.exit()
                else: # dealing with t1amp
                    print('nocc:',eom_obj.nocc)
                    print('operator list t1:',operator_list)
                    print('t1shape:',np.shape(eom_obj.t1amps))
                    a=operator_list[0]-eom_obj.nocc
                    i=operator_list[1]
                    c1[a,i]=amp_key

            if line[:5]=="+++++":#parse the file until this str is read
                read_amps=True


    eom_obj.c2amps=-1.0*c2.transpose(1,3,0,2)  #eom_obj.t2amps.transpose(2,3,1,0)# ijab -> ijba convention ZWW 1/16/25
    eom_obj.c1amps=c1.transpose(1,0)

def read_tensor_info(eom_obj, tei_infile, tamp_infile, eom_infile):
    # different 2e- integral, Tamp, and Ramp storage
    nocc = eom_obj.nocc
    tei = 4.0*return_filled_tensor(tei_infile, 0, "p^ q^ r s",eom_obj)
    eom_obj.tei = tei.transpose(0,1,3,2)

    print("shape of c2amps:",np.shape(eom_obj.c2amps))
   # eom_obj.mp2_energy()
   # print("2e- integral:",eom_obj.tei)
   # sys.exit()
    eom_obj.t1amps = return_filled_tensor(tamp_infile, eom_obj.nocc, "p^ q", eom_obj)
    eom_obj.c2amps = return_filled_tensor(tamp_infile, eom_obj.nocc, "p^ q^ r s", eom_obj)
    sys.exit()
    eom_obj.c1amps = return_filled_tensor(eom_infile, eom_obj.nocc, "p^ q", eom_obj)
    eom_obj.c2amps = return_filled_tensor(eom_infile, eom_obj.nocc, "p^ q r^ s", eom_obj)

    print(eom_obj.c1amps.shape,eom_obj.c2amps.shape, eom_obj.c2amps.shape,eom_obj.t1amps.shape)
    #sys.exit()
    # first transpose T2 and C2 from the current ordering a^ i b^ j to -1.0* a^ b^ i j
    eom_obj.c2amps.transpose(0,2,1,3)
    eom_obj.c2amps.transpose(0,2,1,3)

    eom_obj.c2amps = -1.0*eom_obj.c2amps
    eom_obj.c2amps = -1.0*eom_obj.c2amps 

    # now expand the tensor 
    eom_obj.c2amps = expanded_tensor(eom_obj,eom_obj.c2amps)
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
        print("here is am")
    else:
        if len(index_pattern) == 9:
            pattern = re.compile(
                r"""
    ^\s*
    (?P<indices>(?:\d+\^?\s+)+)   # index tokens like "4^ 5^ 0 1 "
    \|\s*                         # literal pipe separator
                (?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)  # float / scientific notation
                \s*$
                """,
                re.VERBOSE
            )

            print("here i am now")
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
            print("index tokens:", index_tokens) 
            indices = []
            for tok in index_tokens:
                idx = int(tok[:-1]) - offset if tok.endswith("^") else int(tok)
                if idx < 0:
                    raise ValueError(f"Invalid index {tok} with nocc={nocc}")
                indices.append(idx)
    
            value = float(match.group("value"))
            values.append(value)

            op_strings.append(tuple(indices))
            print(indices,value)
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
    #print(op_strings[2])
    # populate tensor
    for inds, val in zip(op_strings,values):
        tensor[inds] = val

    return tensor

