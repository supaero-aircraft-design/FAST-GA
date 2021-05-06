within ;
  
model ICengine

  function compute_sfc_pmax
    input Integer fuel_type;
    input Integer strokes_nb;
    input Real maximum_power;
    output Real sfc_power_max;
  algorithm 
    if fuel_type == 1 then
      if strokes_nb == 2 then
        sfc_power_max := 1125.9 * maximum_power^(-0.2441);
      else
        sfc_power_max := (-0.0011 * maximum_power^2.0) + 0.5905 * maximum_power + 228.58;
      end if;
    else
      if strokes_nb == 2 then
        sfc_power_max := (-0.765 * maximum_power) + 334.94;
      else
        sfc_power_max := (-0.964 * maximum_power) + 231.91;
      end if;
    end if;
  end compute_sfc_pmax;
  parameter Real maximum_power = 1e5 "Maximum power @ a given altitude";
  parameter Integer fuel_type = 2 "fuel type";
  parameter Integer strokes_nb = 4 "engine strokes number";
  parameter Real speed_of_sound = 300.0;
  parameter Real propeller_efficiency = 0.84;
  parameter Real sfc_power_max = compute_sfc_pmax(fuel_type, strokes_nb, maximum_power);
  
  Modelica.Blocks.Math.Gain units_conversion(k=1 / 1e6 / 3600)
    annotation (Placement(visible = true, transformation(extent = {{-36, 2}, {-16, 22}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant sfc_pmax(k = sfc_power_max) annotation(
    Placement(visible = true, transformation(extent = {{-84, 2}, {-64, 22}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant thrust(k = 1000) annotation(
    Placement(visible = true, transformation(extent = {{-134, -38}, {-114, -18}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant mach(k = 0.6) annotation(
    Placement(visible = true, transformation(extent = {{-134, -72}, {-114, -52}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain(k = speed_of_sound) annotation(
    Placement(visible = true, transformation(extent = {{-100, -72}, {-80, -52}}, rotation = 0)));
  Modelica.Blocks.Math.Product product annotation(
    Placement(visible = true, transformation(origin = {-62, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain1(k = 1 / propeller_efficiency) annotation(
    Placement(visible = true, transformation(extent = {{-44, -50}, {-24, -30}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant constant1(k = maximum_power) annotation(
    Placement(visible = true, transformation(extent = {{-46, -86}, {-26, -66}}, rotation = 0)));
  Modelica.Blocks.Math.Division power_ratio annotation(
    Placement(visible = true, transformation(origin = {0, -46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product product1 annotation(
    Placement(visible = true, transformation(origin = {34, -78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain2(k = 1.9964) annotation(
    Placement(visible = true, transformation(extent = {{32, -56}, {52, -36}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain3(k = -0.9976) annotation(
    Placement(visible = true, transformation(extent = {{56, -88}, {76, -68}}, rotation = 0)));
  Modelica.Blocks.Math.Add sfc_ratio annotation(
    Placement(visible = true, transformation(origin = {90, -50}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product sfc annotation(
    Placement(visible = true, transformation(origin = {24, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product dm_dt annotation(
    Placement(visible = true, transformation(origin = {60, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation

  connect(sfc_pmax.y, units_conversion.u) annotation(
    Line(points = {{-62, 12}, {-40, 12}, {-40, 12}, {-38, 12}}, color = {0, 0, 127}));
  connect(mach.y, gain.u) annotation(
    Line(points = {{-113, -62}, {-102, -62}}, color = {0, 0, 127}));
  connect(thrust.y, product.u1) annotation(
    Line(points = {{-113, -28}, {-74, -28}, {-74, -34}}, color = {0, 0, 127}));
  connect(gain.y, product.u2) annotation(
    Line(points = {{-79, -62}, {-79, -46}, {-74, -46}}, color = {0, 0, 127}));
  connect(product.y, gain1.u) annotation(
    Line(points = {{-51, -40}, {-46, -40}}, color = {0, 0, 127}));
  connect(gain1.y, power_ratio.u1) annotation(
    Line(points = {{-23, -40}, {-12, -40}}, color = {0, 0, 127}));
  connect(constant1.y, power_ratio.u2) annotation(
    Line(points = {{-25, -76}, {-18, -76}, {-18, -52}, {-12, -52}}, color = {0, 0, 127}));
  connect(power_ratio.y, product1.u1) annotation(
    Line(points = {{12, -46}, {20, -46}, {20, -72}, {22, -72}}, color = {0, 0, 127}));
  connect(product1.u2, power_ratio.y) annotation(
    Line(points = {{22, -84}, {16, -84}, {16, -46}, {12, -46}, {12, -46}}, color = {0, 0, 127}));
  connect(gain2.u, power_ratio.y) annotation(
    Line(points = {{30, -46}, {12, -46}, {12, -46}, {12, -46}}, color = {0, 0, 127}));
  connect(product1.y, gain3.u) annotation(
    Line(points = {{46, -78}, {52, -78}, {52, -78}, {54, -78}}, color = {0, 0, 127}));
  connect(gain2.y, sfc_ratio.u1) annotation(
    Line(points = {{54, -46}, {76, -46}, {76, -44}, {78, -44}}, color = {0, 0, 127}));
  connect(sfc_ratio.u2, gain3.y) annotation(
    Line(points = {{78, -56}, {74, -56}, {74, -72}, {86, -72}, {86, -80}, {76, -80}, {76, -78}, {78, -78}}, color = {0, 0, 127}));
  connect(units_conversion.y, sfc.u1) annotation(
    Line(points = {{-14, 12}, {10, 12}, {10, 12}, {12, 12}}, color = {0, 0, 127}));
  connect(sfc_ratio.y, sfc.u2) annotation(
    Line(points = {{102, -50}, {114, -50}, {114, -20}, {0, -20}, {0, 0}, {12, 0}, {12, 0}}, color = {0, 0, 127}));
  connect(sfc.y, dm_dt.u1) annotation(
    Line(points = {{36, 6}, {48, 6}}, color = {0, 0, 127}));
  connect(dm_dt.u2, gain1.y) annotation(
    Line(points = {{48, -6}, {-24, -6}, {-24, -40}, {-22, -40}}, color = {0, 0, 127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="3.2.2")));

end ICengine;
