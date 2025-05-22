// Define the component functions object only once
var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

// Graph component 1 - without click handling
dagcomponentfuncs.DCC_Graph1 = function (props) {
    return React.createElement(window.dash_core_components.Graph, {
        figure: props.value,
        style: {height: '100%'},
        config: {
            displayModeBar: false,
            staticPlot: true // This prevents all interactions including clicks
        }
    });
};

// Graph component 2 - without click handling
dagcomponentfuncs.DCC_Graph2 = function (props) {
    return React.createElement(window.dash_core_components.Graph, {
        figure: props.value,
        style: {height: '100%'},
        config: {
            displayModeBar: false,
            staticPlot: true // This prevents all interactions including clicks
        }
    });
};

// Loading overlay component (unchanged)
dagcomponentfuncs.CustomLoadingOverlay = function (props) {
    return React.createElement(
        'div',
        {
            style: {
                display: 'none'
            },
        },
        props.loadingMessage
    );
};