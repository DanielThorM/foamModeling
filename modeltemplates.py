import importlib as imp
import meshmodel as mm
import lsdynakeyword as kw
imp.reload(mm)
imp.reload(kw)
import numpy as np
import os
import subprocess
#def_gradient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.2]])
def periodic_template(tessellation, model_file_name, def_gradient, rho=0.05, phi=0.0, material_data={}, **kwargs):
    options = {
        'elem_type': 16,
        'strain_rate':  1.0,
        'size_coeff': 1.0,
        'strain_coeff': 1.0,
        'n_steps_coeff':500,
        'pert_nodes': 0.0,
        'pert_shell': 0.0,
        'tt_sigma':0.0,
        'csa': 0.0,
        'run': False,
        'return_copy': False,
        'sim_type':'implicit',
        'airbag':False,
        'beam_shape':'straight', #'marvi
        'beam_cs_shape':'round', #'tri'
        'shell_nip':7
    }
    options.update(kwargs)
    material = {
        'e':1500,
        'sigy':25.0,
        'etan':1.0,
        'pr':0.3,
        'ro':9.2e-10,
        'matfail':2.0,
        'soften':False,
        'stress':None,
        'strain':None,
        'rate_c':0.0,
        'rate_p':0.0,
        'rate_mod':None, #[slope1, slope2, base_rate, break_rate], eg. [0.075, 0.2, 1e-3, 3e1]
        'mat_type':'mat24', #'mat181'
        'fs':0.76,
        'fd':0.76,
        'dc':0.0
    }
    material.update(material_data)

    mesh_geometry = mm.FoamModel(tessellation) # mesh_geometry = LSDynaPerGeom(perTessGeometry, debug=True) #mesh_geometry = LSDynaPerGeom(tessellation, debug=True)
    keyword = kw.Keyword(model_file_name)# model_file_name = r'H:\thesis\periodic\representative\S05R1\ID1\testKey.key'
    keyword.comment_block('Control')
    keyword.control_structured()
    endtim = options['strain_coeff']*options['size_coeff'] / options['strain_rate']
    keyword.control_termination(endtim=endtim)
    sampling_number = (options['n_steps_coeff'] * options['size_coeff'] * options['strain_coeff'])
    if options['sim_type'] == 'implicit':
        keyword.control_implicit_general(imflag=1,
                                         dt0=endtim/sampling_number)
        keyword.control_implicit_auto(dtmin=endtim/(sampling_number*100), dtmax=endtim*2/sampling_number, iteopt=25)
        keyword.control_contact(shlthk=2)
        iacc=1
        if options['elem_type']==2:
            iacc=0
        keyword.control_accuracy(osu=1, inn=2, iacc=iacc)
        keyword.control_shell(istupd=4, psstupd=0, irnxx=-2, miter=2, nfail1=1, nfail4=1, esort=2)
        keyword.control_implicit_solution(dctol=5e-5, ectol=5e-4)
        keyword.control_implicit_solver()
        keyword.control_implicit_dynamics()

    else:
        keyword.control_timestep(dt2ms=0.0, tssfac=0.9)
        keyword.control_shell(istupd=4)

    keyword.comment_block('Database')
    keyword.database_glstat(dt=endtim / sampling_number)
    keyword.database_binary_d3_plot(dt=endtim / (30 * options['size_coeff'] * options['strain_coeff']))
    keyword.database_nodout(dt=keyword.endtim / sampling_number)
    keyword.database_nodfor(dt=keyword.endtim / sampling_number)
    keyword.database_spcforc(dt=keyword.endtim / sampling_number)
    keyword.database_bndout(dt=keyword.endtim / sampling_number)

    keyword.comment_block('Material, sections and parts')
    if material['mat_type'] == 'mat24':
        material['e']
        if material['stress'] != None:
            keyword.define_curve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=material['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        elif material['rate_mod'] != None:
            keyword.mat24_rate(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                               etan=material['etan'], ro=material['ro'], pr=material['pr'],
                               str_mod=material['rate_mod'], soften = material['soften'])  # DefaultParams, check LSKey
        elif material['soften']==True:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            keyword.define_curve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=mat_dict['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        else:
            keyword.mat24(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                          etan=material['etan'], pr=material['pr'], c=material['rate_c'], p=material['rate_c'],
                          ro=material['ro'])

    elif material['mat_type'] == 'mat181':
        if material['strain'] == None:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            mat_dict={'strain':list(-1*np.array(mat_dict['strain'])[::-1]) + list(mat_dict['strain'][1:]),
                      'stress':list(-1*np.array(mat_dict['stress'])[::-1]) + list(mat_dict['stress'][1:])}
            keyword.defineCurve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
        else:
            keyword.defineCurve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
        keyword.mat181(mid=1, youngs=mat_dict['E'], ro=material['ro'], lcid=100)

    keyword.mat_null(mid=2, e = material['e'])
    #keyword.mat24(mid=2, e=material['e']/2, sigy=0.0001, fail=material['matfail'], etan=0.00001)


    mesh_geometry.set_csa_sigma(options['csa_sigma'])
    mesh_geometry.set_tt_sigma(options['tt_sigma'])
    mesh_geometry.set_rho(rho=rho, phi=phi)

    if phi != 1.0: # If not only beams
        keyword.element_shell(mesh_geometry.shell_elements)
        if options['elem_type'] == 2 or options['elem_type'] == 10:
            keyword.control_hourglass(ihg=4, qh=0.05)
        elif abs(options['elem_type']) == 16:
            keyword.control_hourglass(ihg=8, qh=0.1)

        for surfs in mesh_geometry.surfs.values():
            keyword.section_shell(secid=surfs.id_, t1=surfs.tt, elform=options['elem_type'], nip=options['shell_nip'])
            if surfs.slave==True:
                keyword.part_contact(pid=surfs.id_, secid=surfs.id_, mid=2,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])
            else:
                keyword.part_contact(pid=surfs.id_, secid=surfs.id_, mid=1,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

    if phi !=0.0:
        mesh_geometry.set_beam_shape(options['beam_shape'])
        if options['beam_cs_shape'] == 'tri':
            keyword.mat28(mid=4, e=material['e'], sigy=material['sigy'], etan=material['etan'], pr=material['pr'],
                          ro=material['ro'])
            keyword.element_beam_section07_orientation(mesh_geometry.beam_elements)
            beam_elform = 2
            for beam in mesh_geometry.beams.values():
                mid=4
                if beam.slave == True:
                    mid=2
                keyword.section_beam(secid=beam.id_, elform=beam_elform)
                keyword.part_contact(pid=beam.id_, secid=beam.id_, mid=mid,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

        elif options['beam_cs_shape'] == 'round':
            keyword.element_beam_thickness_orientation(mesh_geometry.beam_elements)
            beam_elform = 1
            for beam in mesh_geometry.beams.values():
                mid=1
                if beam.slave == True:
                    mid = 2
                keyword.section_beam(secid=beam.id_, csa = beam.csa, elform=beam_elform)
                keyword.part_contact(pid=beam.id_, secid=beam.id_, mid=mid,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])


    if options['sim_type'] == 'implicit':
        #keyword.contact_automatic_single_surface_mortar_id(cid=5200001+i, ssid=5000001+i, sstyp=2, ignore=1)
        keyword.contact_automatic_single_surface_mortar_id(cid=5200001, ssid=0, sstyp=2, ignore=1)
    elif options['sim_type'] == 'explicit':
        keyword.set_part_list(sid=99,
                              pid_list=[solid.id_ for solid in mesh_geometry.solids.values()])
        keyword.contact_automatic_single_surface_id(cid=5200001, ssid=99, sstyp=6,
                                                 ignore=1, igap=2, snlog=1)

    if options['airbag'] == True:
        for i, volume in enumerate(
                mesh_geometry.tessellation.polyhedrons.values()):  # volume = list(mesh_geometry.tessObject.polyhedrons.values())[0]
            keyword.set_part_list(sid=5000001 + i,
                                  pid_list=[abs(surface) * 10 + mesh_geometry.surf_num_offset for surface in
                                            volume.faces])
        keyword.database_abstat(dt=keyword.endtim / sampling_number)
        number_of_polyhedons = i
        for i in range(number_of_polyhedons):
            keyword.airbag_adiabatic_gas_model(abid=5100001 + i, sid=5000001 + i)
    ##########################################################################
    if options['pert_nodes'] != 0.0:
        keyword.pertubation_node(options['pert_nodes'], nsid=0, cmp=1, xwl=mesh_geometry.tessellation.domain_size[0]/10,
                                 ywl=0, zwl=0)
    if options['pert_shell'] != 0.0:
        keyword.pertubationShell(options['pert_shell'], nsid=0, cmp=1, xwl=mesh_geometry.tessellation.domain_size[0]/10,
                                 ywl=0, zwl=0)
    ##########################################################################



    #def_gradient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.2]])
    keyword.node(mesh_geometry.nodes)
    BCs = kw.BoundaryConditions(keyword, mesh_geometry)
    if options['sim_type'] == 'implicit':
        keyword.disp_type = 'disp'

    elif options['sim_type'] == 'explicit':
        keyword.disp_type = 'vel'
        keyword.mat24(mid=3, e=1e7, ro=1e-7, pr=0.0, sigy=1e9)
        keyword.element_solid(mesh_geometry.solid_elements)
        keyword.section_solid(secid=3, elform=2)
        for solid in mesh_geometry.solids.values():
            keyword.part(pid=solid.id_, secid=3, mid=3)

    if options['return_copy']==True:
       return BCs


    if options['sim_type'] == 'implicit':
        keyword = BCs.periodic_linear_local(def_gradient)
        keyword.end_key()
        keyword.write_key()
    elif options['sim_type'] == 'explicit':
        keyword = BCs.periodic_multiple_global(def_gradient)
        keyword.end_key()
        keyword.write_key()
##########################################################################

##########################################################################
    ##########################################################################
    if options['run'] == True: #Move to keyword_file?
        rem_working_folder = os.getcwd()
        os.chdir(keyword.model_file_name.rsplit('\\',1)[0])
        SOLVER = r'C:\Program Files\LSTC\LS-DYNA\ls-dyna_smp_d_R10.0_winx64_ifort160.exe'
        INPUT = model_file_name
        MEMORY = str(50)
        NCPU = str(1)
        subprocess.Popen('"{}" i={} ncpu={} memory={}m'.format(SOLVER, INPUT, NCPU, MEMORY))
        os.chdir(rem_working_folder)
    #return readASCI.readrwforc(workingFolder=workingFolder)

periodic_template(tessellation, 'testcsa_sigma.key', def_gradient, csa_sigma=0.1, tt_sigma=0.1)