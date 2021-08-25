pure function adjust_alpha_noncollinear(spin_in, spin_plus_in, target_spin, alpha) result(alpha_opt)
    use prec
    implicit none
    real(q), intent(in) :: spin_in(:, :)
    real(q), intent(in) :: spin_plus_in(:, :)
    real(q), intent(in) :: target_spin(:, :)
    real(q), intent(in) :: alpha
    real(q) :: alpha_opt
    real(q) :: sum_k, sum_k2
    sum_k = sum((target_spin - spin_in)*(spin_plus_in - spin_in))
    sum_k2 = sum((spin_in - spin_plus_in)**2)
    alpha_opt = sum_k*alpha/sum_k2
end function adjust_alpha_noncollinear
