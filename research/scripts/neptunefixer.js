function makeTableColored(table) {
    var rows = table.getElementsByTagName("tr");
    for (i = 0; i < rows.length; i++) {
        // three colors: white, light grey, light blue
        if (i % 3 == 2) {
            rows[i].style.backgroundColor = "white";
        } else if (i % 3 == 1) {
            rows[i].style.backgroundColor = "#e6e6e6";
        } else {
            rows[i].style.backgroundColor = "#e6f2ff";
        }
    }
}

// sort table according to first value. Keep the first 2 rows (header) unchanged
function sortTable(table) {
    var rows = table.getElementsByTagName("tr");
    var rowsArray = Array.prototype.slice.call(rows, 2);
    rowsArray.sort(function (a, b) {
        var aVal = a.getElementsByTagName("td")[0].getAttribute("title");
        var bVal = b.getElementsByTagName("td")[0].getAttribute("title");
        // return aVal - bVal;
        // reverse order
        return bVal - aVal;
    });
    for (i = 0; i < rowsArray.length; i++) {
        table.appendChild(rowsArray[i]);
    }
}

// global variable "issorting" set to False. This is to prevent coloring tables (therefore
// changing the DOM) while sorting them, because it would cause the sorting to be broken.
var issorting = false;

function makeAllTablesColored() {
    // don't execute if issorting is True
    if (issorting) return;
    var tables = document.getElementsByTagName("table");
    // filter only tables with class "compare-experiments-chart-tooltip__table"
    tables = Array.prototype.filter.call(tables, function (table) {
        return table.classList.contains("compare-experiments-chart-tooltip__table");
    });
    for (i = 0; i < tables.length; i++) {
        makeTableColored(tables[i]);
    }
}


function makeAllTablesSorted() {
    issorting = true;
    var tables = document.getElementsByTagName("table");
    // filter only tables with class "compare-experiments-chart-tooltip__table"
    tables = Array.prototype.filter.call(tables, function (table) {
        return table.classList.contains("compare-experiments-chart-tooltip__table");
    });
    for (i = 0; i < tables.length; i++) {
        sortTable(tables[i]);
    }
    issorting = false;
    makeAllTablesColored();
}

// execute it every time an element is created
document.addEventListener("DOMNodeInserted", makeAllTablesColored);
// execute it every 2 seconds as well
// setInterval(makeAllTablesGrey, 100);

// when "s" is pressed, sort all tables
document.addEventListener("keypress", function (e) {
    if (e.key == "s") {
        makeAllTablesSorted();
    }
});

// // when "a" is pressed, make all tables grey
// unnecessary, because it is executed every time an element is created
// document.addEventListener("keypress", function (e) {
//     if (e.key == "a") {
//         makeAllTablesGrey();
//     }
// });
