
// 3D matrix class for transformers.
class Matrix3D {

private:

    // The matrix.
    float*** mMat;

    // Matrix dimensionality in rows, columns, depth.
    int *mDims;

    // Allocates the matrix.
    void AllocateMat() {
        int i, j;
        mMat = new float**[mDims[0]];
        for(i = 0; i < mDims[0]; ++i) {
            mMat[i] = new float*[mDims[1]];
            for(j = 0; j < mDims[1]; ++j)
                mMat[i][j] = new float[mDims[2]];
        }
    }

    // Deallocates the matrix.
    void DeallocateMat() {
        int i, j;
        for(i = 0; i < mDims[0]; ++i) {
            for(j = 0; j < mDims[1]; ++j)
                delete[] mMat[i][j];
            delete mMat[i];
        }
        delete[] mMat;
    }

    Matrix3D *TransposeDepthToRows() {
        int height = mDims[2];
        int width = mDims[1];
        int depth = mDims[0];
        Matrix3D *transposed = new Matrix3D(height, width, depth);
        int i, j, k;
        // For each row of new matrix...
        for(i = 0; i < height; ++i)
            // For each column of each row of new matrix...
            for(j = 0; j < width; ++j) 
                // For each index of each column of each row of matrix...
                for(k = 0; k < depth; ++k) 
                    transposed->SetValue(i, j, k, this->GetValue(k, j, i));
        return transposed;
    }

    Matrix3D *TransposeDepthToColumns() {
        int height = mDims[0];
        int width = mDims[2];
        int depth = mDims[1];
        Matrix3D *transposed = new Matrix3D(height, width, depth);
        int i, j, k;
        // For each row of new matrix...
        for(i = 0; i < height; ++i)
            // For each column of each row of new matrix...
            for(j = 0; j < width; ++j) 
                // For each index of each column of each row of matrix...
                for(k = 0; k < depth; ++k) 
                    transposed->SetValue(i, j, k, this->GetValue(i, k, j));
        return transposed;
    }  

    Matrix3D *TransposeRowsToColumns() {
        int height = mDims[1];
        int width = mDims[0];
        int depth = mDims[2];
        Matrix3D *transposed = new Matrix3D(height, width, depth);
        int i, j, k;
        // For each row of new matrix...
        for(i = 0; i < height; ++i)
            // For each column of each row of new matrix...
            for(j = 0; j < width; ++j) 
                // For each index of each column of each row of matrix...
                for(k = 0; k < depth; ++k) 
                    transposed->SetValue(i, j, k, this->GetValue(j, i, k));
        return transposed;
    }  

public:

    enum TransposeMode { 
        depthToRows, 
        depthToColumns,
        rowsToColumns
    };

    Matrix3D(int x, int y, int z) {
        mDims = new int[3];
        mDims[0] = x;
        mDims[1] = y;
        mDims[2] = z;
        AllocateMat();
    }
    ~Matrix3D() {
        DeallocateMat();
        delete mDims;
    }

    int GetHeight() {
        return mDims[0];
    }

    int GetWidth() {
        return mDims[1];
    }

    int GetDepth() {
        return mDims[2];
    }

    void SetValue(int x, int y, int z, float val) {
        mMat[x][y][z] = val;
    }

    float GetValue(int x, int y, int z) {
        return mMat[x][y][z];
    }

    Matrix3D *Transpose(TransposeMode mode) {
        Matrix3D *transposed = nullptr;
        switch(mode) {
            case depthToRows:
                transposed = TransposeDepthToRows();
                break;
            case depthToColumns:
                transposed = TransposeDepthToColumns();
                break;
            case rowsToColumns:
                transposed = TransposeRowsToColumns();
                break;
        }
        return transposed;
    }

    // Multiplies two 3D matricies, assumes they have the same shape (or you're in big trouble!).
    static Matrix3D *Multiply(Matrix3D *a, Matrix3D *b, int axisA, int axisB) {
        return nullptr;
    }
};