from flask import Flask, render_template, request,g
import sympy
from sympy import symbols, Eq, solve, Integral, Derivative, Matrix, lambdify
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.plotting import plot
import pickle
import math
from scipy import stats
from scipy.stats import binom
import statistics as stats
from scipy.special import comb
import sqlite3
from datetime import datetime as dt
app = Flask(__name__)


# Create a dictionary to store previously solved problems
prev_problems = {}
DATABASE = 'database.db'
# Define a function to create a new database connection and cursor object
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = sqlite3.connect(DATABASE)
        g._database = db
    return db

# Define a function to create the comments table in the database if it doesn't exist
def create_table():
    c = get_db()
    c.execute("CREATE TABLE IF NOT EXISTS comments (name TEXT, comment TEXT, date TEXT)")
    #c.close()

# Define a function to add user comments to the database
def add_comment(name, comment):
    now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO comments (name, comment, date) VALUES (?, ?, ?)", (name, comment, now))
    conn.commit()
    c.close()
    return "Comment added succesfully"


# Define a function to get all comments from the database
def get_comments():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM comments ORDER BY date DESC")
    comments = cur.fetchall()
    cur.close()
    return comments


# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    explanation = None
    comments = []
    create_table()
    if request.method == 'POST':
        # Get the values from the form fields
        problem = request.form['problem']
        name = request.form.get('name')
        comment = request.form.get('comment')

        # Check that the required fields are not empty
        if name and comment:
            # Add the comment to the database
            add_comment(name, comment)
            
        if problem in prev_problems:
            explanation = prev_problems[problem]
        else:
            explanation = solve_problem(problem)
            prev_problems[problem] = explanation

        # Get all comments from the database
        comments = get_comments()
        
        return render_template('results.html', problem=problem, explanation=explanation, comments=comments)
    else:
        return render_template('index.html')
        


# Close the database connection
def close_db():
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
# Define the route for the results page
@app.route('/result')
def result():
    problem = request.args.get('problem')
    explanation = solve_problem(problem)
    return render_template('results.html', problem=problem, explanation=explanation)
    
def solve_problem(problem_str):
    solutions = None
    problem_str = problem_str.replace(" ", "")
    # Check if the problem has been solved before
    try:
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
            
            # Add explanation and examples
            explanation = f"To solve an equation, we want to find the values of the variable that make the equation true. For example, consider the equation {problem_str}. To solve it, we can rearrange it to get all the terms with the variable on one side and all the constants on the other side. Then we can solve for the variable. For instance, given the equation {problem_str}, we can start by subtracting {right} from both sides to get {left} = 0. Then we can solve for x using the solve() function in SymPy. For instance, solve({left}, x) = {solutions}."
            more_examples = f""" \n [NOTE]
                As a machine , I am not yet trained to show you all the steps to solve your problem but , I can give you an idea of how equations are solved by most humans. 
                Example 1:  2^2(x+3) + 9 - 5 = 32 
                [step1]. Resolve the exponent. Remember the order of operations: PEMDAS, which stands for Parentheses, Exponents, Multiplication/Division, and Addition/Subtraction.[1] You can't resolve the parentheses first because x is in the parentheses, so you should start with the exponent, 2^2. 2^2 = 4 , therefore: 4(x+3) + 9 - 5 = 32], 
                [step2].Do the multiplication: Just distribute the 4 into (x +3). Here's how:4x + 12 + 9 - 5 = 32 
                [step3]. Do The addition and Subtraction: Just add or subtract the remaining numbers. Here's how:4x+21-5 = 32 ,4x+16 = 32, 4x + 16 - 16 = 32 - 16, 4x = 16 
                [step4]. Isolate the Variable:To do this, just divide both sides of the equation by 4 to find x. 4x/4 = x and 16/4 = 4, so x = 4.4x/4 = 16/4,x = 4. 
                [step5].Now Check Your work: Just plug x = 4 back into the original equation to make sure that it checks out. Here's how:22(x+3)+ 9 - 5 = 32,22(4+3)+ 9 - 5 = 32,22(7) + 9 - 5 = 32,4(7) + 9 - 5 = 32,28 + 9 - 5 = 32,37 - 5 = 32,32 = 32
                
                
                \n Example 2:    Let's say you're working with this problem where the x term includes an exponent:
                    2x2 + 12 = 44
              
                [step2]

                Isolate the term with the exponent.[5] The first thing you should do is combine like terms so that all of the constant terms are on the right side of the equation while the term with the exponent is on the left side. Just subtract 12 from both sides. Here's how:
                    2x2+12-12 = 44-12
                    2x2 = 32


                [step3]

                Isolate the variable with the exponent by dividing both sides by the coefficient of the x term. In this case, 2 is the x coefficient, so divide both sides of the equation by 2 to get rid of it. Here's how:
                    (2x2)/2 = 32/2
                    x2 = 16

                [step4]

                Take the square root of each side of the equation.[6] Taking the square root of x2 will cancel it out. So, take the square root of both sides. You'll get x left over on one side and plus or minus the square root of 16, 4, on the other side. Therefore, x = ±4.

                [step5]

                Check your work. Just plug x = 4 and x = -4 back in to the original equation to make sure it checks out. For instance, when you're checking x=4:
                    2x2 + 12 = 44
                    2 x (4)2 + 12 = 44
                    2 x 16 + 12 = 44
                    32 + 12 = 44
                    44 = 44

            Another Method:

            (Using Fractions)

                [step1]
                Write down the problem. Let's say you're working with the following problem:[7]
                    (x + 3)/6 = 2/3

                [step2]

                Cross multiply. To cross multiply, simply multiply the denominator of each fraction by the numerator of the other fraction.[8] You'll be essentially multiplying in two diagonal lines. So, multiply the first denominator, 6, by the second numerator, 2, to get 12 on the right side of the equation. Multiply the second denominator, 3, by the first numerator, x + 3, to get 3 x + 9 on the left side of the equation. Here's how it will look:
                    (x + 3)/6 = 2/3
                    6 x 2 = 12
                    (x + 3) x 3 = 3x + 9
                    3x + 9 = 12

                [step3]
                Combine like terms. Combine the constant terms in the equation to subtract 9 from both sides of the equation.[9] Here's what you do:
                    3x + 9 - 9 = 12 - 9
                    3x = 3

                [step4]
                Isolate x by dividing each term by the x coefficient. Just divide 3x and 9 by 3, the x term coefficient, to solve for x. 3x/3 = x and 3/3 = 1, so you're left with x = 1.[10]
                Image titled Solve for X Step 16
                [step5]
                Check your work. To check your work, just plug x back in to the original equation to make sure that it works.[11] Here's what you do:
                    (x + 3)/6 = 2/3
                    (1 + 3)/6 = 2/3
                    4/6 = 2/3
                    2/3 = 2/3

                """
        # Check if the problem is a system of equations
        elif 'system' in problem_str:
            eqns = problem_str.split(',')
            eqns = [sympy.parsing.sympy_parser.parse_expr(eqn) for eqn in eqns]
            vars = []
            for eqn in eqns:
                vars += list(eqn.free_symbols)
            vars = sorted(set(vars), key=lambda var: str(var))
            solutions = sympy.solve(eqns, vars)
            
            # Add explanation and examples
            explanation = f"A system of equations is a collection of equations with multiple variables. To solve a system of equations, we want to find the values of the variables that make all the equations true. For example, consider the system of equations {problem_str}. To solve it, we can use the solve() function in SymPy. For instance, given the system {problem_str}, we can parse each equation using parse_expr(), find the variables using free_symbols(), sort the variables alphabetically, and solve the system using solve(). For instance, solve({eqns}, {vars}) = {solutions}."
            more_examples=f""" \n Here are three methods that humans use to solve system equations:

                                Graphing Method: This method involves graphing the equations on a coordinate plane and identifying the point(s) where the graphs intersect. For example, consider the system of equations:

                                x + y = 5
                                2x - y = 4

                                Graphing these equations on a coordinate plane will result in two lines that intersect at the point (2, 3), which is the solution to the system.

                                Substitution Method: This method involves solving one of the equations for one of the variables, and then substituting that expression into the other equation. For example, consider the system of equations:

                                x + y = 5
                                2x - y = 4

                                Solving the first equation for y results in y = 5 - x. Substituting this expression for y into the second equation yields:

                                2x - (5 - x) = 4

                                Solving for x, we get x = 3. Substituting this value into the first equation yields:

                                3 + y = 5

                                Solving for y, we get y = 2. So the solution to the system is x = 3 and y = 2.

                                Elimination Method: This method involves adding or subtracting the equations in such a way that one of the variables is eliminated, and then solving for the remaining variable. For example, consider the system of equations:

                                2x + 3y = 7
                                4x - y = 11

                                Multiplying the second equation by 3 and adding it to the first equation yields:

                                14x = 40

                                Solving for x, we get x = 40/14, which simplifies to x = 20/7. Substituting this value into the second equation yields:

                                4(20/7) - y = 11

                                Solving for y, we get y = -13/7. So the solution to the system is x = 20/7 and y = -13/7."""

        # Check if the problem is a polynomial equation
        elif 'poly' in problem_str:
            expr = sympy.parsing.sympy_parser.parse_expr(problem_str.replace('poly', ''))
            solutions = sympy.roots(expr, multiple=True)
            
            # Add explanation and examples
            explanation = f"A polynomial equation is an equation where the variable appears with integer powers only. To solve a polynomial equation, we want to find the values of the variable that make the equation true. For example, consider the polynomial equation {problem_str}. To solve it, we can use the roots() function in SymPy. For instance, given the equation {problem_str}, we can parse it using parse_expr() and find its roots using roots(). For instance, roots({expr}, multiple=True) = {solutions}."
            more_examples=f""" \n Two common methods that humans use to solve polynomial equations are:

                                Factoring Method: In this method, the polynomial equation is factored into simpler expressions, and the roots are determined from the factors. For example, consider the polynomial equation: x^2 - 5x + 6 = 0. We can factor this as (x - 3)(x - 2) = 0, which gives the roots x = 3 and x = 2.

                                Quadratic Formula: This is a formula that gives the roots of a quadratic equation, which is a polynomial equation of degree 2. The formula is: x = (-b ± sqrt(b^2 - 4ac)) / 2a, where a, b, and c are the coefficients of the quadratic equation. For example, consider the quadratic equation: x^2 - 5x + 6 = 0. Using the quadratic formula, we have a = 1, b = -5, and c = 6, so x = (5 ± sqrt(25 - 24)) / 2 = 3 or 2.

                                Note that these methods can also be used to solve other types of polynomial equations, but the degree of the polynomial and the complexity of the coefficients will affect the difficulty of the solution."""
        #Check if the problem involves trigonometric functions
        elif 'sin' in problem_str or 'cos' in problem_str or 'tan' in problem_str:
            # Split the problem into function and angle
            function, angle = problem_str.split('(')[0], problem_str.split('(')[1].split(')')[0]
            
            # Calculate the result based on the function and angle
            if function == 'sin':
                result = round(math.sin(math.radians(float(angle))), 2)
                solutions = result
            elif function == 'cos':
                result = round(math.cos(math.radians(float(angle))), 2)
                solutions = result
            elif function == 'tan':
                result = round(math.tan(math.radians(float(angle))), 2)
                solutions = result
            
            # Add explanation and examples
            explanation = f"To calculate the {function} of {angle} degrees, we first convert {angle} degrees to radians, and then apply the {function} function to the result. For instance, {function}({angle}) = {result}."
            more_examples= f""" \n Two methods for determining trigonometric functions using algebraic equations, graphs, and more advanced mathematical techniques are:  

                                 Algebraic method: This method involves the use of algebraic equations and identities to determine trigonometric functions. For example, the Pythagorean identity, sin^2(x) + cos^2(x) = 1, can be used to determine the value of sin(x) or cos(x) if the value of the other trigonometric function is known. Similarly, the sum and difference identities can be used to find the value of a trigonometric function in terms of other trigonometric functions.

                                Graphical method: This method involves the use of graphs to determine trigonometric functions. Trigonometric functions can be graphed on a coordinate plane using the unit circle or by using a graphing calculator. By analyzing the shape and position of the graph, the values of the trigonometric function can be determined for specific angles. For example, the sine function has a maximum value of 1 at 90 degrees and a minimum value of -1 at 270 degrees, and the cosine function has a maximum value of 1 at 0 degrees and a minimum value of -1 at 180 degrees.

                            More advanced mathematical techniques, such as calculus and complex analysis, can also be used to determine trigonometric functions. For example, the derivatives and integrals of trigonometric functions can be used to find their values at specific points, and complex analysis can be used to express trigonometric functions in terms of exponential functions."""
        # Check if the problem is a derivative
        elif 'derivative' in problem_str:
            expr = sympy.parsing.sympy_parser.parse_expr(problem_str.replace('derivative', ''))
            solutions = sympy.solve(expr.diff())
            
            # Add explanation and examples
            explanation = f"A derivative is a measure of how a function changes as its input changes. To find the derivative of a function, we differentiate the function with respect to its input variable. For example, consider the derivative {problem_str}. To find the derivative, we can differentiate the function using the diff() function in SymPy. For instance, given the derivative {problem_str}, we can parse it using parse_expr() and find its derivative using diff(). For instance, diff({expr}) = {solutions}."
            more_examples = f""" \n Two common methods used by humans to determine a derivative are:

                                        Using the definition of the derivative: The definition of the derivative is the limit of the difference quotient as the change in the independent variable approaches zero. This method involves finding the limit of the difference quotient using algebraic manipulation and substitution. For example, to find the derivative of f(x) = x^2 at x = 2, we can use the definition of the derivative:

                                    f'(2) = lim h->0 [(2+h)^2 - 2^2] / h
                                    = lim h->0 [4h + h^2] / h
                                    = lim h->0 [4 + h] = 4

                                    Therefore, f'(2) = 4.

                                        Using differentiation rules: Differentiation rules are a set of algebraic rules that allow us to find derivatives of functions more easily. The most common differentiation rules include the power rule, product rule, quotient rule, chain rule, and trigonometric rules. For example, to find the derivative of f(x) = 3x^4 - 2x^3 + 5x^2 - 6x + 1, we can apply the power rule and linearity of differentiation:

                                    f'(x) = d/dx [3x^4] - d/dx [2x^3] + d/dx [5x^2] - d/dx [6x] + d/dx [1]
                                    = 12x^3 - 6x^2 + 10x - 6

                                    Therefore, f'(x) = 12x^3 - 6x^2 + 10x - 6."""  
                                    
                                    
        # Check if the problem is to calculate square root
        elif '√' in problem_str:
            num_str = problem_str.split('√')[1].strip()
            num = float(num_str)
            solutions = math.sqrt(num)
            
            # Add explanation and examples
            explanation = f"To find the square root of a number, we can use the math.sqrt() function in Python's math module. For example, to find the square root of {num}, we can use math.sqrt({num}) = {solutions}."
            more_examples = f""" \n Two common methods used by humans to find the square root of a number are:

                                        Using the division method: The division method involves grouping the digits of the number in pairs, starting from the decimal point and moving left. We then find the largest digit whose square is less than or equal to the first pair of digits. We write this digit as the first digit of the square root and subtract its square from the first pair of digits. We then bring down the next pair of digits and repeat the process until all pairs of digits have been exhausted. For example, to find the square root of 7.29 using the division method, we can proceed as follows:

                                        Step 1: Group the digits in pairs, starting from the decimal point and moving left: 7|29
                                        Step 2: Find the largest digit whose square is less than or equal to 7: 2
                                        Step 3: Write 2 as the first digit of the square root: √7 = 2...
                                        Step 4: Subtract 4 from 7: 7 - 4 = 3
                                        Step 5: Bring down the next pair of digits: 29
                                        Step 6: Double the first digit of the current result (2) to get 4, and write it as the divisor: 2|29
                                        Step 7: Find the largest digit whose product with the divisor is less than or equal to 329: 4
                                        Step 8: Write 4 as the next digit of the square root: √7.29 = 2.7...

                                        Using the long division method: The long division method involves dividing the number whose square root we want to find by a number that gives a close approximation of the square root, and repeating the process until the desired accuracy is obtained. For example, to find the square root of 7.29 using the long division method, we can proceed as follows:

                                        Step 1: Guess a number whose square is less than or equal to 7.29, such as 2.
                                        Step 2: Divide 7.29 by 2 to get 3.645.
                                        Step 3: Average 2 and 3.645 to get 2.8225.
                                        Step 4: Divide 7.29 by 2.8225 to get 2.5830...
                                        Step 5: Average 2.8225 and 2.5830 to get 2.70275...
                                        Step 6: Repeat steps 4 and 5 until the desired accuracy is obtained: √7.29 ≈ 2.70275..."""

                      
        # Check if the problem is a partial differential equation with more than two variables
        elif 'wave' in problem_str:
            x, y, z, t = symbols('x y z t')
            u = sympy.Function('u')(x, y, z, t)
            eq = Eq(sympy.diff(u, t, t) - sympy.diff(u, x, x) - sympy.diff(u, y, y) - sympy.diff(u, z, z), 0)
            solutions = sympy.pde.pdsolve(eq)
            # Add explanation and examples
            explanation = f"The wave equation is a partial differential equation that describes the behavior of waves in a medium. To solve a partial differential equation like the wave equation, we can use the pdsolve() function in SymPy. For example, consider the wave equation {eq}. To solve this equation, we can use the pdsolve() function in SymPy. For instance, pdsolve({eq}) = {solutions}."
            more_examples=''
        # Check if the problem is a simple addition or subtraction
        elif '+' in problem_str or '-' in problem_str:
            # Split the problem into operands and calculate the result
            result = eval(problem_str)
            
            # Add explanation and examples
            operator = '+' if '+' in problem_str else '-'
            operands = problem_str.split(operator)
            solutions = result
            explanation = f"To calculate the result of {problem_str}, we {operator} the operands {operands[0]} and {operands[1]}. For instance, {operands[0]} {operator} {operands[1]} = {result}."
            more_examples=''
            
        elif 'P(' in problem_str or 'C(' in problem_str or 'mean' in problem_str or 'median' in problem_str or 'mode' in problem_str:
            # Split the problem into type and arguments
            problem_type = problem_str.split('(')[0]
            arguments = problem_str.split('(')[1].split(')')[0].split(',')
        
            # Calculate the result based on the problem type
            if problem_type == 'P':
                p_success = float(arguments[0])
                n_trials = int(arguments[1])
                n_successes = int(arguments[2])
                result = round(binom.pmf(n_successes, n_trials, p_success), 4)
                solutions=result
                
            elif problem_type == 'C':
                n_items = int(arguments[0])
                k_choices = int(arguments[1])
                result = round(comb(n_items, k_choices), 4)
                solutions=result
                
            elif problem_type == 'mean':
                values = [float(x) for x in arguments]
                result = round(stats.mean(values), 2)
                solutions=result
                
            elif problem_type == 'median':
                values = [float(x) for x in arguments]
                result = round(stats.median(values), 2)
                solutions=result
                
            elif problem_type == 'mode':
                values = [float(x) for x in arguments]
                result = stats.mode(values)
                solutions=result
                
            # Add explanation and examples
            explanation = f"To calculate the {problem_type} of the given data, we apply the appropriate formula or function. For instance, {problem_str} = {result}."
            more_examples=''
        else:
            raise ValueError("Invalid problem type")

        # Save the solution for future reference
        prev_problems[problem_str] = (solutions, explanation)

        return (f"The answer is:{solutions}   , {explanation},{more_examples}")

    except Exception as e:
        return (None, f"Error: {str(e)}")


    except SyntaxError as e:
        # if there's a syntax error, return the error message and the problematic input string
        return f"SyntaxError: {e}\nProblematic input string: {problem_str}"

    except Exception as e:
        # if there's any other error, return an error message
        return f"Error: {e}"


   
    
if __name__=='__main__':
    app.run(debug=True)