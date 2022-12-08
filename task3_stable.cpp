//
// Created by general on 27.11.22.
//

// Мб, MPI течет- bash build_and_run.sh joke 5 5 1 debug
// https://github.com/open-mpi/ompi/issues/10048
// Мои матрицы удаляются- с ними и без них одни и те же утечки

#include <mpi.h>
#include <vector>
#include <iostream>
#include <tuple>
#include <unistd.h>
#include <cmath>
#include <iomanip>


const double A1 = -2;
const double A2 = 3;
const double B1 = -1;
const double B2 = 4;
const double eps = 5e-6;

enum CART_DIMENSION {
    Y_DIM = 0,
    X_DIM
};

enum IDX {
    LEFT = 0,
    RIGHT,
    TOP,
    BOTTOM
};

double u(double x, double y)
{
    return 2.0 / (1.0 + x*x + y*y);
}

double psi_r(double x, double y)
{
    return (-4.0 * x * (1.0 + (x+y) * (x+y))) / ((1.0 + x*x + y*y) * (1.0 + x*x + y*y));
}

inline double psi_l(double x, double y)
{
    return (4.0 * x * (1.0 + (x+y) * (x+y))) / ((1.0 + x*x + y*y) * (1.0 + x*x + y*y));
}

inline double psi_t(double x, double y)
{
    return (-4.0 * y * (1.0 + (x+y) * (x+y))) / ((1.0 + x*x + y*y) * (1.0 + x*x + y*y));
}

inline double psi_b(double x, double y)
{
    return (4.0 * y * (1.0 + (x+y) * (x+y))) / ((1.0 + x*x + y*y) * (1.0 + x*x + y*y));
}

inline double k(double x, double y)
{
    return 1.0 + (x+y) * (x+y);
}

inline double q(double x, double y)
{
    return 1.0;
}

inline double F(double x, double y)
{
    return (8.0 * (x*x + 4*x*y + y*y + 1.0)) / (( x*x + y*y + 1.0)*( x*x + y*y + 1.0)*( x*x + y*y + 1.0))
                + 2.0 / (1.0 + x*x + y*y);
}

struct Matrix2D
{
    // TODO: possible replace with vector
    double* data{nullptr};
    int n_rows, n_cols;
    // because it's offset of (1,1) element, edge rows and cols are "virtual", core matrix in (1,1)->(n_rows-2, n_cols-2)
    int global_row_offset_11, global_col_offset_11;

    Matrix2D() = default;

    Matrix2D(int num_rows, int num_cols, int row_gl_offset_of_11, int col_gl_offset_of_11)
        : n_rows(num_rows), n_cols(num_cols), global_row_offset_11(row_gl_offset_of_11), global_col_offset_11(col_gl_offset_of_11)
    {
        data = new double[n_rows * n_cols]();
    }

    void resetMatrix2D(int num_rows, int num_cols, int row_gl_offset_of_11, int col_gl_offset_of_11) {
        if (data != nullptr) {
            delete[] data;
            data = nullptr;
        }
        n_rows = num_rows;
        n_cols = num_cols;
        global_row_offset_11 = row_gl_offset_of_11;
        global_col_offset_11 = col_gl_offset_of_11;
        data = new double[n_rows * n_cols]();
    }

    inline int ind(int row, int col) {
        return row * n_cols + col;
    }

    inline double& el(int row, int col) {
        return data[row * n_cols + col];
    }

    void printWholeMatrix() {
        printf("x_offset= {%d}, y_offset= {%d} \n", global_col_offset_11, global_row_offset_11);
        fflush(stdout);
        for (int row = n_rows-1 ; row >= 0; --row) {
            for (int col = 0; col < n_cols; ++col) {
                std::cout << std::setw(15) << std::setfill(' ') << std::setprecision(10) << std::left << data[row * n_cols + col] << ' ';
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }
    }

    void printCoreMatrixAndInfo() {
        printf("x_offset= {%d}, y_offset= {%d} \n", global_col_offset_11, global_row_offset_11);
        fflush(stdout);
        for (int row = n_rows - 2; row > 0; --row) {
            printf("row= {%d}: ", row);
            for (int col = 1; col < n_cols - 1; ++col) {
                printf("%.10f ", data[row * n_cols + col]);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }
    }

    void swap(Matrix2D& other) {
        if (!(other.n_rows == this->n_rows && other.n_cols == this->n_cols))  {
            fprintf(stderr, "wrong swap!!!");
            throw 1;
        }
        double* tmp_ptr = other.data;
        other.data = this->data;
        this->data = tmp_ptr;
    }

    ~Matrix2D() {
        // std::cout << "dtor ";
        delete[] data;
        data = nullptr;
    }
};

class Process {
public:
    int num_procs, my_rank_world;

    Process(int M_x_dots_total /*cols*/, int N_y_dots_total /*rows*/, double eps_in) {
        eps_to_beat = eps_in;

        CreateCommunicator(M_x_dots_total, N_y_dots_total);

        W_row_col.resetMatrix2D(y_size + 2, x_size + 2, row_global_offset, col_global_offset);
        W_row_col_prev.resetMatrix2D(y_size + 2, x_size + 2, row_global_offset, col_global_offset);

        // need to fill this matrices
        a_row_col.resetMatrix2D(y_size + 2, x_size + 3, row_global_offset, col_global_offset);
        b_row_col.resetMatrix2D(y_size + 3, x_size + 2, row_global_offset, col_global_offset);
        B_row_col.resetMatrix2D(y_size + 2, x_size + 2, row_global_offset, col_global_offset);
        r_row_col.resetMatrix2D(y_size + 2, x_size + 2, row_global_offset, col_global_offset);
        Ar_row_col.resetMatrix2D(y_size + 2, x_size + 2, row_global_offset, col_global_offset);
        q_row_col.resetMatrix2D(y_size + 2, x_size + 2, row_global_offset, col_global_offset);

        W_left_prev_recv.resize(y_size+2); W_left_send.resize(y_size+2);          // (y_size + 2)
        W_right_prev_recv.resize(y_size+2); W_right_send.resize(y_size+2);        // (y_size + 2)
        W_top_prev_recv.resize(x_size+2); W_top_send.resize(x_size+2);            // (x_size + 2)
        W_bottom_prev_recv.resize(x_size+2); W_bottom_send.resize(x_size+2);      // (x_size + 2)

        shadow_bufs_paires_ptrs[SEND] = W_left_send.data();
        shadow_bufs_paires_ptrs[RECV] = W_right_prev_recv.data();
        shadow_bufs_paires_ptrs[2+SEND] = W_right_send.data();
        shadow_bufs_paires_ptrs[2+RECV] = W_left_prev_recv.data();
        shadow_bufs_paires_ptrs[4+SEND] = W_top_send.data();
        shadow_bufs_paires_ptrs[4+RECV] = W_bottom_prev_recv.data();
        shadow_bufs_paires_ptrs[6+SEND] = W_bottom_send.data();
        shadow_bufs_paires_ptrs[6+RECV] = W_top_prev_recv.data();
    }

    std::pair<int, double> finite_diffs_method() {       // метод конечных разностей
        double eps_global = 1000.0;
        // TODO: need to test filled values!!! Especially at corners
        // tested in hpc3_funcs
        FillGridsWithInitValues();
//        for (int i = 0; i < num_procs; i++) {
//            MPI_Barrier(MPI_COMM_WORLD);
//            if (my_rank_world == i) {
//                B_row_col.printCoreMatrixAndInfo();
////                std::cout << "rank= " << my_rank_world << ": has_neighbours: lrtb" << std::endl;
////                for (int i = 0; i < 4; ++i) {
////                    std:: cout << hn[i] << " ";
////                }
////                std::cout << std::endl;
//            }
//            sleep(1);
//        }
//        return {};
//        if (my_rank_world == 0) {
//            std::cout << "rank= " << my_rank_world << ": has_neighbours: lrtb" << std::endl;
//            for (int i = 0; i < 4; ++i) {
//                std:: cout << hn[i] << " ";
//            }
//        }
        int iteration_number = 0;
        double start_time = 0;
        double all_procs_communication_time = 0;
        double this_proc_calculation_time = 0;
        double this_proc_total_time = 0;
        // SubstituteSolution();
        do {
            start_time = MPI_Wtime();
            W_row_col_prev.swap(W_row_col);

            double s1 = MPI_Wtime();
            ExchangeData(); // обмен теневыми гранями
            all_procs_communication_time += MPI_Wtime() - s1;

            s1 = MPI_Wtime();
            finite_diffs_iteration();
            this_proc_calculation_time += MPI_Wtime() - s1;

            // выполнение редукционной операции
            s1 = MPI_Wtime();
            MPI_Allreduce(&norms_local,&norms_global,3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            all_procs_communication_time += MPI_Wtime() - s1;
//            std::cout << "Ar_r = " << norms_global[Ar_r_prod] << " ,Ar_Ar= " << norms_global[Ar_square] << " ,r_r= " << norms_global[r_square] << std::endl;

            s1 = MPI_Wtime();
            tau = norms_global[Ar_r_prod] / norms_global[Ar_square];
            eps_global = std::abs(tau) * sqrt(norms_global[r_square]);
            MakeStep();
            this_proc_calculation_time += s1 - MPI_Wtime();

            this_proc_total_time += MPI_Wtime() - start_time;
            if (my_rank_world == 0 && (iteration_number % 1000 == 0)) {
                printf("# %d : eps_global= %.7f ,tau= %.10f ,communication time= %.10f ,calculations time= %.10f ,total time= %.10f\n",
                       iteration_number, eps_global, tau, all_procs_communication_time, this_proc_calculation_time, this_proc_total_time);
//                std::cout << "r: " << std::endl;
//                r_row_col.printWholeMatrix();
//                std::cout << "Ar: " << std::endl;
//                Ar_row_col.printWholeMatrix();
            }
            if (iteration_number % 1000 == 0) {
                this_proc_calculation_time = 0;
                all_procs_communication_time = 0;
                this_proc_total_time = 0;
            }

            ++iteration_number;
            if (iteration_number > 1e7) {
                break;
            }
        } while(eps_global > eps_to_beat);

        if (my_rank_world == 0) {
            std::cout << "# " << iteration_number << ": eps_global= " << eps_global << " ,tau= " << tau << "  <--- FINAL" << std::endl;
//            std::cout << "r: " << std::endl;
//            r_row_col.printWholeMatrix();
//            std::cout << "B: " << std::endl;
//            B_row_col.printWholeMatrix();
        }
        return std::make_pair(iteration_number, eps_global);
    }

    void PrintAllData() {
//        for (int i = 0; i < num_procs; i++) {
//            MPI_Barrier(MPI_COMM_WORLD);
//            if (my_rank_world == i    ) {
//
//            }
//        }


//        if ((num_procs < 4 && my_rank_world == 0) || (num_procs > 3 && my_rank_world == 3)) {
        printf("Dims = [{%d}, {%d}]\n", dims[Y_DIM], dims[X_DIM]);
        // FIXME: need to check разбивку- что везде правильное количество точек и правильные отступы
        std::cout << "orig_y_coord= " << my_cartezian_coords[Y_DIM] << ", orig_x_coord= " << my_cartezian_coords[X_DIM]
                  << ", grid_y_coord= " << my_grid_coords[Y_DIM] << ", grid_x_coord= " << my_grid_coords[X_DIM] << " ";
        std::cout << "x_size= " << x_size << ", y_size= " << y_size << '\n';
        std::cout << "y_offset= " << W_row_col.global_row_offset_11 << ", x_offset= " << W_row_col.global_col_offset_11 << std::endl;
        // W_row_col.printCroppedMatrixAndInfo();
        // W_row_col.printWholeMatrix();
        usleep(5*1000);
    }

private:
    int y_size /*rows*/, x_size /* cols */;
    double hx, hy;
    double eps_to_beat;                 //точность
    // omp_lock_t dmax_lock;

    MPI_Comm cart_comm;
    int my_cartezian_coords[2];
    int my_grid_coords[2];
    int row_global_offset, col_global_offset;
    int dims[2] = {0};          // количество процессов в каждом измерении

    bool hn[4] = {false, false, false, false};  // has neighbour LEFT, RIGHT, TOP, BOTTOM

    // TODO: почти все есть в лекции 3!!!!!
    // TODO: трюк со сменой w_cur и w_prev
    //текущее значение блока и буфер с предыдущим,
    Matrix2D W_row_col, W_row_col_prev;             // (y_size + 2) * (x_size + 2)
    Matrix2D Ar_row_col;                            // (y_size + 2) * (x_size + 2)
    Matrix2D q_row_col;                             // (y_size + 2) * (x_size + 2)
    Matrix2D B_row_col;                             // (y_size + 2) * (x_size + 2)
    Matrix2D r_row_col;                             // (y_size + 2) * (x_size + 2)
    double tau = 0;

    enum BUF_IDX {
        SEND = 0,
        RECV
    };
    //  буферы для формирования сообщений
    double *shadow_bufs_paires_ptrs[8];
    std::vector<double> W_left_prev_recv, W_left_send;          // (y_size + 2)
    std::vector<double> W_right_prev_recv, W_right_send;        // (y_size + 2)
    std::vector<double> W_top_prev_recv, W_top_send;            // (x_size + 2)
    std::vector<double> W_bottom_prev_recv, W_bottom_send;      // (x_size + 2)

    Matrix2D a_row_col;                             // (y_size + 2) * (x_size + 3)
    Matrix2D b_row_col;                             // (y_size + 3) * (x_size + 2)

    enum NORMS {
        Ar_r_prod = 0,
        Ar_square,
        r_square
    };

    double norms_local[3] = {0.0, 0.0, 0.0};   // Arr_sq_norm_and_ArAr_sq_norm_and_r_norm_local
    double norms_global[3] = {0.0, 0.0, 0.0}; // Arr_sq_norm_and_ArAr_sq_norm_and_r_norm_global

    double laplace_edge_aware(int row, int col, Matrix2D& W_row_col_lap,
                              const double& w_row_col_minus1, const double& w_row_col_plus1,
                              const double& w_row_minus1_col, const double& w_row_plus1_col) {
        // 0 <= row <= y_size + 1
        // 0 <= col <= x_size + 1

        // W_row_col: (y_size + 2) * (x_size + 2)
        // q_row_col: (y_size + 2) * (x_size + 2)

        //  a_row_col: (y_size + 2) * (x_size + 3), но формально это матрица y_size * (x_per_col + 1), тк сначала строки, потом колонки
        // hint on a_row_col: x_per_col + 1 тк хотим еще точка с индексом x_size + 2, чтобы считать с правой стороны 5 точечный шаблон (те 1 лишние колонка, тк x_per_col- это от 0 до x_size + 1) (чтобы можно было 5-точечную схему на краях считать)

        // b_row_col: (y_size + 3) * (x_size + 2)
        // y_size + 3 тк там 1 лишних строка для элемента y_size + 2, который нужен во 2 производной для y_size + 1

        if ((col == 0 && !hn[LEFT]) || (col == (x_size+1) && !hn[RIGHT]) || (row == (y_size+1) && !hn[TOP]) || (row == 0 && !hn[BOTTOM])) {
            // нет даже смысла считать точку, которая "за" (левее, правее, выше, ниже) гранью
            return 0;
        }
//        if ((col > 0 && col < W_row_col_lap.n_cols - 1) && (row > 0 && row < W_row_col_lap.n_rows - 1)) {
//            // fill params from W_row_col_lap
//        }

        double w_row_col = W_row_col_lap.el(row, col);

        bool only_first_der_x = (col == 1 && !hn[LEFT]) || (col == x_size && !hn[RIGHT]);
        bool only_first_der_y = (row == y_size && !hn[TOP]) || (row == 1 && !hn[BOTTOM]);

        bool x_term_positive = (col == x_size && !hn[RIGHT]); //  Если x в последнем столбце и это граничное условие, то тогда слагаемое со знаком + входит в граничных условиях.
        bool y_term_positive = (row == y_size && !hn[TOP]);	 // Если y на последней строке и это граничное условие, то тогда слагаемое со знаком + входит в граничных условиях

        double x_term = 0.0;
        if (only_first_der_x) {
            // Если не нужна 2 производная, то это значит, что не нужно читать из буффера по этому направлению, и можно прям руками взять и достать из W_row_col_lap значения. Это удобно при отступах на левой границе и на нижней границе
            int add_col = (col == 1 && !hn[LEFT]); // нужно ли отступить от края слева ? add_col = 1 => only_first_der_x == true
            double w_row_col_add = W_row_col_lap.el(row, col + add_col);
            double w_row_col_minus1_add = W_row_col_lap.el(row, col + add_col - 1);
            x_term = (2.0 / hx) * a_row_col.el(row, col + add_col) * (w_row_col_add - w_row_col_minus1_add) / hx;
        } else {
            // Если по направлению 2 производных, то очень может быть, что мы читаем одно из значений соседних из буфера
            double a_wx_line = a_row_col.el(row, col) * (w_row_col - w_row_col_minus1) / hx;
            x_term = a_row_col.el(row, col + 1) * (w_row_col_plus1 - w_row_col) / (hx * hx) - a_wx_line / hx; // test with row = y_size + 1 && col = x_size + 1->(y_size + 2) * (x_size + 3) - 1
        }

        double y_term = 0.0;
        if (only_first_der_y) {
            int add_row = (row == 1 && !hn[BOTTOM]);
            double w_row_add_col = W_row_col_lap.el(row + add_row, col);
            double w_row_minus1_add_col = W_row_col_lap.el(row + add_row - 1, col);
            y_term = (2.0 / hy) * b_row_col.el(row + add_row, col) * (w_row_add_col - w_row_minus1_add_col) / hy;
        } else {
            double b_wy_line = b_row_col.el(row, col) * (w_row_col - w_row_minus1_col) / hy;
            // TODO: сделать так, чтобы соседние строки рядом лежали
            y_term = b_row_col.el(row + 1, col) * (w_row_plus1_col - w_row_col) / (hy * hy) - b_wy_line / hy; // test with row = y_size + 1 && col = x_size + 1->(y_size + 3) * (x_size + 2) - 1
        }

        if (!x_term_positive) {
            x_term *= -1;
        }

        if (!y_term_positive) {
            y_term *= -1;
        }

        // tested that Au - B = 0 in hpc_3.py
        double res = x_term + y_term + q_row_col.el(row, col) * w_row_col;
        return res;
    }

    void laplace_core_inplace(int row, int col) {
        if (row == 0 || row == W_row_col.n_rows-1 || col == 0 || col == W_row_col.n_cols-1) {
            return;
        }
        r_row_col.el(row, col) = -(a_row_col.el(row, col + 1) * (W_row_col_prev.el(row, col+1) - W_row_col_prev.el(row, col)) / (hx * hx)
                - a_row_col.el(row, col) * (W_row_col_prev.el(row, col) - W_row_col_prev.el(row, col-1)) / (hx * hx)
                + b_row_col.el(row + 1, col) * (W_row_col_prev.el(row+1, col) - W_row_col_prev.el(row, col)) / (hy * hy)
                - b_row_col.el(row, col) * (W_row_col_prev.el(row, col) - W_row_col_prev.el(row-1, col)) / (hy * hy))
                        + q_row_col.el(row, col) * W_row_col_prev.el(row, col) - B_row_col.el(row, col);
    }

    void CalculateR() {

        int nRows = W_row_col_prev.n_rows;
        int nCols = W_row_col_prev.n_cols;

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < nCols; ++col) {

                r_row_col.el(row,col) =  laplace_edge_aware(row, col, W_row_col_prev,
                                                            col != 0 ? W_row_col_prev.el(row, col-1) : W_left_prev_recv[row],
                                                            col != nCols-1 ? W_row_col_prev.el(row, col+1) : W_right_prev_recv[row],
                                                            row != 0 ? W_row_col_prev.el(row-1, col) : W_bottom_prev_recv[col],
                                                            row != nRows-1 ? W_row_col_prev.el(row+1, col) : W_top_prev_recv[col]);
                r_row_col.el(row,col) -= B_row_col.el(row,col);
            }
        }

        for (int row = 3; row < nRows - 3; ++row) {
            int col = 0;
            for (col = 0; col < 2; ++col) {
                r_row_col.el(row, col) = laplace_edge_aware(row, col, W_row_col_prev,
                                                            col != 0 ? W_row_col_prev.el(row, col - 1)
                                                                     : W_left_prev_recv[row],
                                                            col != nCols - 1 ? W_row_col_prev.el(row, col + 1)
                                                                             : W_right_prev_recv[row],
                                                            row != 0 ? W_row_col_prev.el(row - 1, col)
                                                                     : W_bottom_prev_recv[col],
                                                            row != nRows - 1 ? W_row_col_prev.el(row + 1, col)
                                                                             : W_top_prev_recv[col])
                                            - B_row_col.el(row, col);
            }
            for (col = 2; col < nCols - 2; ++col) {
                laplace_core_inplace(row, col);
            }
            for (col = nCols - 2; col < nCols; ++col) {
                r_row_col.el(row, col) = laplace_edge_aware(row, col, W_row_col_prev,
                                                            col != 0 ? W_row_col_prev.el(row, col - 1)
                                                                     : W_left_prev_recv[row],
                                                            col != nCols - 1 ? W_row_col_prev.el(row, col + 1)
                                                                             : W_right_prev_recv[row],
                                                            row != 0 ? W_row_col_prev.el(row - 1, col)
                                                                     : W_bottom_prev_recv[col],
                                                            row != nRows - 1 ? W_row_col_prev.el(row + 1, col)
                                                                             : W_top_prev_recv[col]) -
                                         B_row_col.el(row, col);
            }
        }

        for (int row = nRows - 3; row < nRows; ++row) {
            for (int col = 0; col < nCols; ++col) {
                r_row_col.el(row,col) = laplace_edge_aware(row, col, W_row_col_prev,
                                                            col != 0 ? W_row_col_prev.el(row, col-1) : W_left_prev_recv[row],
                                                            col != nCols-1 ? W_row_col_prev.el(row, col+1) : W_right_prev_recv[row],
                                                            row != 0 ? W_row_col_prev.el(row-1, col) : W_bottom_prev_recv[col],
                                                            row != nRows-1 ? W_row_col_prev.el(row+1, col) : W_top_prev_recv[col]) - B_row_col.el(row,col);
            }
        }
    }

    void add_norms(int row, int col, int nRows, int nCols) {
        double ro_ij = (col == nCols - 2 || col == 1 ? 0.5 : 1) * (row == nRows - 2 || row == 1 ? 0.5 : 1);
        norms_local[Ar_r_prod] += hx * hy * Ar_row_col.el(row, col)  * r_row_col.el(row, col) * ro_ij;
        norms_local[Ar_square] += hx * hy * Ar_row_col.el(row, col)  * Ar_row_col.el(row, col) * ro_ij;
        norms_local[r_square] += hx * hy * r_row_col.el(row, col) * r_row_col.el(row, col) * ro_ij;
    }

    void CalculateTauParts() {
        int nRows = W_row_col_prev.n_rows;
        int nCols = W_row_col_prev.n_cols;
        norms_local[Ar_r_prod] = 0;
        norms_local[Ar_square] = 0;
        norms_local[r_square] = 0;
        for (int row = 1; row < nRows-1; ++row) {
            for (int col = 1; col < nCols-1; ++col) {
                Ar_row_col.el(row, col) = laplace_edge_aware(row, col, r_row_col,
                                                             r_row_col.el(row, col - 1), r_row_col.el(row, col + 1),
                                                             r_row_col.el(row - 1, col), r_row_col.el(row + 1, col));
                add_norms(row, col, nRows, nCols);
            }
        }
    }

    void finite_diffs_iteration() {             // шаг метода конечных разностей

        CalculateR();
        CalculateTauParts();

        return;
    }

    void CreateCommunicator(int M_x_dots_total, int N_y_dots_total) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_world);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Dims_create(num_procs, 2, dims);

        int periods[2] = {false, false};

        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);
        MPI_Cart_coords(cart_comm, my_rank_world, 2, my_cartezian_coords);

        // initially M = 100 ->  M_x_dots = 101 -> hx = (A2 - A1) / (M_x_dots - 1)
        hy = (B2 - B1) / (N_y_dots_total - 1);
        hx = (A2 - A1) / (M_x_dots_total - 1);

        int procs_per_rows = dims[Y_DIM];
        // Такая топология на полюсе проверена руками
        // first n_procs_with_extra_y_dots get extra_y_dot      | -----------> x
        //                                                      V y
        int n_procs_with_extra_y_dot = N_y_dots_total % procs_per_rows;
        y_size = N_y_dots_total / procs_per_rows + int(my_cartezian_coords[Y_DIM] < n_procs_with_extra_y_dot);

        int procs_per_columns = dims[X_DIM];
        int n_procs_with_extra_x_dot = M_x_dots_total % procs_per_columns;
        x_size = M_x_dots_total / procs_per_columns + int(my_cartezian_coords[X_DIM] < n_procs_with_extra_x_dot);

        my_grid_coords[Y_DIM] = dims[Y_DIM] - 1 - my_cartezian_coords[Y_DIM];  // row coord         ^ y
        my_grid_coords[X_DIM] = my_cartezian_coords[X_DIM];                     // col coord        | -----> x

        // Я проверил топологию- оно так и есть и на полюсе и на локальной машине!
        hn[LEFT] = my_grid_coords[X_DIM] != 0;
        hn[RIGHT] = my_grid_coords[X_DIM] != (dims[X_DIM] - 1);
        hn[TOP] = my_grid_coords[Y_DIM] != (dims[Y_DIM] - 1);
        hn[BOTTOM] = my_grid_coords[Y_DIM] != 0;

        int n_procs_wo_extra_y_dot = procs_per_rows - n_procs_with_extra_y_dot;
        row_global_offset = (N_y_dots_total / procs_per_rows) * (my_grid_coords[Y_DIM])
                            + (my_grid_coords[Y_DIM] < n_procs_wo_extra_y_dot ? 0 : my_grid_coords[Y_DIM] - n_procs_wo_extra_y_dot);

        col_global_offset = (M_x_dots_total / procs_per_columns) * (my_grid_coords[X_DIM])
                            + (my_grid_coords[X_DIM] < n_procs_with_extra_x_dot ? my_grid_coords[X_DIM] : n_procs_with_extra_x_dot);
    }

    void FillGridsWithInitValues() {
        double start_x = A1 + (col_global_offset - 1) * hx;
        double start_y = B1 + (row_global_offset - 1) * hy;
        double cur_x, cur_y;

        for (int row = 0; row < y_size + 2; ++ row) {
            cur_y = start_y + row * hy;
            cur_x = start_x;
            for (int col = 0; col < x_size + 2; ++col) {
                if (row == y_size + 1) {
                    // fiil b_row_col[y_size+3][col]
                    b_row_col.el(row+1, col) = k(cur_x, cur_y + 0.5 * hy);
                }
                if (col == x_size + 1) {
                    // fiil a_row_col[row][x_size+2]
                    a_row_col.el(row, col+1) = k(cur_x + 0.5 * hx, cur_y);
                }
                a_row_col.el(row, col) = k(cur_x - 0.5 * hx, cur_y);
                b_row_col.el(row, col) = k(cur_x,cur_y - 0.5 * hy);
                q_row_col.el(row, col) = 1;
                B_row_col.el(row, col) = F(cur_x, cur_y);
                r_row_col.el(row, col) = 1e5;
                if (row > 0 && row < y_size+1 && col > 0 && col < x_size+1) {
                    if (row == 1 && !hn[BOTTOM]) {
                        B_row_col.el(row, col) += 2.0 * psi_b(cur_x, cur_y) / hy;
                    } else if (row == y_size && !hn[TOP]) {
                        B_row_col.el(row, col) += 2.0 * psi_t(cur_x, cur_y) / hy;
                    } else if (col == 1 && !hn[LEFT]) {
                        B_row_col.el(row, col) += 2.0 * psi_l(cur_x, cur_y) / hx;
                    } else if (col == x_size && !hn[RIGHT]) {
                        B_row_col.el(row, col) += 2.0 * psi_r(cur_x, cur_y) / hx;
                    }
                }
                cur_x += hx;
            }
        }
        // FIXME: ТОЛЬКО Я СО ВСЕГО КУРСА ЗАМЕТИЛ ОШИБКУ!!!!
        if (!hn[BOTTOM] && !hn[LEFT]) {
            cur_x = start_x + hx;
            cur_y = start_y + hy;
            B_row_col.el(1, 1) = F(cur_x, cur_y)
                    + (2.0/hx + 2.0/hy) * 1.0 / (hx + hy) * (hx * psi_b(cur_x, cur_y)  + hy * psi_l(cur_x, cur_y));
        }
        if (!hn[BOTTOM] && !hn[RIGHT]) {
            cur_x = start_x + x_size* hx;
            cur_y = start_y + hy;
            B_row_col.el(1, x_size) = F(cur_x, cur_y)
                    + (2.0/hx + 2.0/hy) * 1.0 / (hx + hy) * (hx * psi_b(cur_x, cur_y)  + hy * psi_r(cur_x, cur_y));
        }
        if (!hn[TOP] && !hn[LEFT]) {
            cur_x = start_x + hx;
            cur_y = start_y + y_size * hy;
            B_row_col.el(y_size, 1) = F(cur_x, cur_y)
                    + (2.0/hx + 2.0/hy) * 1.0 / (hx + hy) * (hx * psi_t(cur_x, cur_y)  + hy * psi_l(cur_x, cur_y));
        }
        if (!hn[TOP] && !hn[RIGHT]) {
            cur_x = start_x + x_size * hx;
            cur_y = start_y + y_size * hy;
            B_row_col.el(y_size, x_size) = F(cur_x, cur_y)
                    + (2.0/hx + 2.0/hy) * 1.0 / (hx + hy) * (hx * psi_t(cur_x, cur_y) + hy * psi_r(cur_x, cur_y));
        }

    }

    void SubstituteSolution() {
        double start_x = A1 + (col_global_offset - 1) * hx;
        double start_y = B1 + (row_global_offset - 1) * hy;
        double cur_x, cur_y;

        for (int row = 0; row < y_size + 2; ++ row) {
            cur_y = start_y + row * hy;
            cur_x = start_x;
            for (int col = 0; col < x_size + 2; ++col) {
                W_row_col.el(row, col) = u(cur_x, cur_y);
                cur_x += hx;
            }
        }
    }

    void ExchangeData() {

        int nCols = W_row_col.n_cols;
        int nRows = W_row_col.n_rows;


        MPI_Status status;
//          FIXME: TOOOOOOOOOO SLOOOOOOOOW
        int rank_recv, rank_send;
        int dir = 0;
        int disp = 0;
        // std::cout << "Entered data exchange!" << std::endl;
        for (int i = 0; i < 4; ++i) {
            dir = 1 - i / 2; disp = i == 0 || i == 3 ? -1 : 1;
            MPI_Cart_shift(cart_comm, dir, disp, &rank_recv, &rank_send);
            // std::cout << my_rank_world << " " << hn[0] << hn[1] << hn[2] << hn[3] << std::endl;
            if (hn[(1-dir)*2] && hn[(1-dir)*2+1]) {
                // std::cout << my_rank_world << " sendrecv: send to " << rank_send << " ,recv from " << rank_recv << std::endl;
                MPI_Sendrecv(shadow_bufs_paires_ptrs[i*2+SEND], 1-dir ? nCols : nRows, MPI_DOUBLE,
                             rank_send, 27, shadow_bufs_paires_ptrs[i*2+RECV], 1-dir ? nCols : nRows, MPI_DOUBLE,
                             rank_recv, 27, cart_comm, &status);
            } else if (hn[(1-dir)*2 +std::max(0,disp)]) {
                // std::cout << my_rank_world << " send to " << rank_send << std::endl;
                MPI_Send(shadow_bufs_paires_ptrs[i*2+SEND], 1-dir ? nCols : nRows, MPI_DOUBLE,
                        rank_send, 27, cart_comm);
            } else if (hn[(1-dir)*2 + std::abs(std::min(0,disp))]) {
                // std::cout << my_rank_world << " recv from " << rank_recv << std::endl;
                MPI_Recv(shadow_bufs_paires_ptrs[i*2+RECV], 1-dir ? nCols : nRows, MPI_DOUBLE, rank_recv,
                         27, cart_comm, &status);
            }
            // MPI_Barrier(MPI_COMM_WORLD);
        }

    }

    void MakeStep() {
        int nRows = W_row_col_prev.n_rows;
        int nCols = W_row_col_prev.n_cols;

        for (int row = 0; row < nRows; ++row) {
            for (int col = 0; col < nCols; ++col) {
                W_row_col.el(row, col) = W_row_col_prev.el(row, col) - tau * r_row_col.el(row, col);
                if (col == 2 && hn[LEFT]) {
                    W_left_send[row] = W_row_col.el(row, col);
                }
                if (col == nCols-3 && hn[RIGHT]) {
                    W_right_send[row] = W_row_col.el(row, col);
                }
                if (row == 2 && hn[TOP]) {
                    W_bottom_send[col] = W_row_col.el(row, col);
                }
                if (row == nRows-3 && hn[TOP]) {
                    W_top_send[col] = W_row_col.el(row, col);
                }
            }
        }

    }
};

std::pair<int, int> AnalyseCommandString(int argc, char* argv[]) {
    if (argc < 3 && argv[3] == nullptr) {
        fprintf(stderr,"M and N missing\n");
        throw std::runtime_error("M and N are missing");
    }

    int M = -1;
    int N = -1;
    sscanf(argv[1], "%d", &M);
    sscanf(argv[2], "%d", &N);
    if (M < 0 || N < 0) {
        fprintf(stderr,"invalid M or N\n");
        throw std::runtime_error("invalid M or N");
    }
    return std::make_pair(M, N);
}

int main(int argc, char* argv[]) {

    int M = 0;
    int N = 0;
    std::tie(M, N) = AnalyseCommandString(argc, argv);

    MPI_Init(&argc, &argv);
    Process P(M+1, N+1, eps);

    // P.PrintAllData();
    double time1, time2;
    time1 = MPI_Wtime();
    int num_iterations=-1; double accuracy=-1;
    auto res = std::pair<int, double>(num_iterations, accuracy);
    res = P.finite_diffs_method();
    time2 = MPI_Wtime();
    // P.PrintAllData();
    if (P.my_rank_world == 0)  {
        std::cout << "M= " << M << " N= " << N << " ,Num_procs= " << P.num_procs << " ,Num_iterations= " << res.first << " ,Accuracy: " << res.second
                    << " ,Time: " << time2 - time1 << std::endl;
    }

    MPI_Finalize();
    return 0;
}
