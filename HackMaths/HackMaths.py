import sympy
from sympy import symbols, Eq, solve, Integral, Derivative, Matrix, lambdify
from sympy.plotting import plot

# Dictionary to store previously solved problems
prev_problems = {}

#Define function for solving problems
def solve_problem(problem_str):
    problem_str = problem_str.replace(" ", "")
    # Check if the problem has been solved before
    if problem_str in prev_problems:
        return prev_problems[problem_str]

    # Check if the problem is an equation
    if "=" in problem_str:
        # Parse the equation
        left, right = problem_str.split("=")
        expr = sympy.parsing.sympy_parser.parse_expr(left) - sympy.parsing.sympy_parser.parse_expr(right)

        # Solve the equation
        solutions = solve(expr, symbols('x'))

        # If the left-hand side was not zero, add the right-hand side to the solutions
        if left != str(solutions[0]):
            solutions = [sol + sympy.parsing.sympy_parser.parse_expr(right) for sol in solutions]

    # Check if the problem is a system of equations
    elif 'system' in problem_str:
        eqns = problem_str.split(',')
        eqns = [sympy.parsing.sympy_parser.parse_expr(eqn) for eqn in eqns]
        vars = []
        for eqn in eqns:
            vars += list(eqn.free_symbols)
        vars = sorted(set(vars), key=lambda var: str(var))
        solutions = sympy.solve(eqns, vars)

    # Check if the problem is a polynomial equation
    elif 'polynomial' in problem_str:
        expr = sympy.parsing.sympy_parser.parse_expr(problem_str.replace('polynomial', ''))
        solutions = sympy.roots(expr, multiple=True)
        
    # Check if the problem is a derivative
    elif 'derivative' in problem_str:
        expr, x, a = problem_str.split(',')
        expr = sympy.parsing.sympy_parser.parse_expr(expr)
        a = float(a)
        solutions = sympy.diff(expr, symbols(x)).subs(x, a)

    # Check if the problem is an integral
    elif 'integral' in problem_str:
        expr, x, a, b = problem_str.split(',')
        expr = sympy.parsing.sympy_parser.parse_expr(expr)
        a = float(a)
        b = float(b)
        solutions = sympy.integrate(expr, (symbols(x), a, b))
    
    # Check if the problem is a Fourier series
    elif 'fourier series' in problem_str:
        expr, x, a, b = problem_str.split(',')
        expr = sympy.parsing.sympy_parser.parse_expr(expr)
        a = float(a)
        b = float(b)
        period = b - a
        fs = sympy.fourier_series(expr, (symbols(x), a, b))
        solutions = fs.truncate(n=5)

    # Check if the problem is a partial differential equation
    elif 'heat equation' in problem_str:
        x, y, t = symbols('x y t')
        u = sympy.Function('u')(x, y, t)
        k = symbols('k')
        eq = Eq(sympy.diff(u, t) - k*(sympy.diff(u, x, x) + sympy.diff(u, y, y)), 0)
        solutions = sympy.pde.pdsolve(eq)

    else:
        # Parse the problem into a SymPy expression
        expr = sympy.parsing.sympy_parser.parse_expr(problem_str)

        # Solve the problem by simplifying the expression
        solutions = sympy.simplify(expr)


    # Check for additional functionality
    if 'sin' in problem_str or 'cos' in problem_str:
        solutions = sympy.solveset(expr, symbols('x'))
    elif 'tan' in problem_str:
        expr = sympy.parsing.sympy_parser.parse_expr(problem_str)
        solutions = sympy.simplify(expr)
        
    elif 'limit' in problem_str:
        expr, x, a = problem_str.split(',')
        expr = sympy.parsing.sympy_parser.parse_expr(expr)
        a = float(a)
        solutions = sympy.limit(expr, symbols(x), a)
        
    elif 'second derivative' in problem_str:
        expr = problem_str.replace('second derivative of ', '')
        expr = sympy.parsing.sympy_parser.parse_expr(expr)
        solutions = sympy.diff(expr, symbols('x'), 2)
        
    elif 'heat equation' in problem_str:
        u = sympy.Function('u')(symbols('x'), symbols('y'))
        eq = sympy.Eq(sympy.diff(u, symbols('x'), symbols('x')) + sympy.diff(u, symbols('y'), symbols('y')), 0)
        solutions = sympy.pde.pdsolve(eq)
        
    # Check if the problem is a second-order differential equation
    elif 'second order differential equation' in problem_str:
        # Parse the equation
        expr, x = symbols('y'), symbols('x')
        y = sympy.Function('y')(x)
        eq = sympy.Eq(sympy.diff(y, x, x) + 2*sympy.diff(y, x) + 5*y, 0)
        solutions = sympy.dsolve(eq)
        
    # Check if the problem is a partial differential equation with more than two variables
    elif 'wave equation' in problem_str:
        x, y, z, t = symbols('x y z t')
        u = sympy.Function('u')(x, y, z, t)
        eq = Eq(sympy.diff(u, t, t) - sympy.diff(u, x, x) - sympy.diff(u, y, y) - sympy.diff(u, z, z), 0)
        solutions = sympy.pde.pdsolve(eq)
        
    elif 'plot' in problem_str:
        # Parse the function and plot
        expr = sympy.parsing.sympy_parser.parse_expr(problem_str.replace('plot ', ''))
        p = plot(expr, show=False)
        p.show()
        return "Graph displayed."
        
    elif 'numeric' in problem_str:
        # Parse the function, convert to a lambda function, and evaluate numerically
        expr = sympy.parsing.sympy_parser.parse_expr(problem_str.replace('numeric ', ''))
        f = lambdify(x, expr)
        a, b = [float(x) for x in input("Enter the start and end points: ").split()]
        n = int(input("Enter the number of steps: "))
        x_vals = [a + i*(b-a)/n for i in range(n+1)]
        y_vals = [f(x) for x in x_vals]
        p = plot(expr, (x, a, b), show=False)
        p.scatter(x_vals, y_vals)
        p.show()
        return "Graph displayed."
    else:
        pass


    # Store the solution in the previous problems dictionary
    prev_problems[problem_str] = solutions

    # Return the solution(s) to the user
    return solutions

# Main loop
while True:
    # Get user input
    user_input = input("Enter a mathematical problem: ")

    # Check if the problem has already been solved
    if user_input in prev_problems:
        print(f"Result (from history): {prev_problems[user_input]}")
        continue

    # Try to solve the problem
    try:
        solutions = solve_problem(user_input)
        print(f"Result: {solutions}")
    except:
        print("Error: Could not solve the problem.")
