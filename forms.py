from flask_wtf import FlaskForm
from wtforms import DecimalField, IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class ProductPredictForm(FlaskForm):
    NetRevenue = DecimalField('Net Revenue', places=2, validators=[DataRequired()])
    NetQuantity = DecimalField('Net Quantity', places=0, validators=[DataRequired()])
    NumTransactions = IntegerField('Num Transactions', validators=[DataRequired(), NumberRange(min=0)])
    NumUniqueCustomers = IntegerField('Num Unique Customers', validators=[DataRequired(), NumberRange(min=0)])
    NetRevenue_LastMonth = DecimalField('Net Revenue (Last Month)', places=2, validators=[DataRequired()])
    NetRevenue_MA3 = DecimalField('Net Revenue (MA3)', places=2, validators=[DataRequired()])
    Month = IntegerField('Month', validators=[DataRequired(), NumberRange(min=1, max=12)])
    ProductFrequency = IntegerField('Product Frequency', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Run Predictions')