import os
import dash
from options.options import Options

from dashs import create_layout, callbacks

opt = Options()

app = dash.Dash(
    __name__,

)


server = app.server
app.layout = create_layout(app)
callbacks(app)

# Running server
if __name__ == "__main__":
    app.run_server(debug=True)