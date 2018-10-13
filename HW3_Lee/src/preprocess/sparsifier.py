from scipy.sparse import csr_matrix

"""
    Converts a term document matrix into a compressed sparse row (csr) matrix.

    Implementation based on CSR Matrix example from SciPy
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

"""
def sparsifyTermDocument(matrix):
    docs = matrix
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in docs:
        for termId, termCount in d:
            # map each word to a unique id (incrementing idx)
            index = vocabulary.setdefault(termId, len(vocabulary)) 
            # extend the array depending on the termCount 
            indices.extend([index] * termCount) 
            data.extend([1] * termCount)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), dtype=int)